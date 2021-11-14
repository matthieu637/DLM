import collections
import copy
import functools
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io

from keras.utils.np_utils import to_categorical
from difflogic.cli import format_args
from difflogic.dataset.utils import ValidActionDataset
from difflogic.nn.baselines import MemoryNet
from difflogic.nn.critic import *
from difflogic.nn.neural_logic import InputTransform, LogicInference, LogicMachine, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.dlm.layer import DifferentiableLogicMachine
from difflogic.nn.dlm.neural_logic import DLMInferenceBase
from difflogic.nn.rl.reinforce import REINFORCELoss, REINFORCELogLoss

from difflogic.nn.rl.ppo import PPOLoss
from difflogic.train import MiningTrainerBase

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.logging import set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from difflogic.train.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda
from jactorch.utils.meta import as_numpy
from jactorch.utils.meta import as_tensor
from difflogic.tqdm_utils import tqdm_for

TASKS = ['final', 'stack', 'nlrl-Stack', 'nlrl-Unstack', 'nlrl-On', 'sort', 'path']

parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='dlm',
    choices=['nlm', 'memnet', 'dlm'],
    help='model choices, nlm: Neural Logic Machine, memnet: Memory Networks, dlm: Differentiable Logic Machine')

# NLM parameters, works when model is 'nlm'.
nlm_group = parser.add_argument_group('Neural Logic Machines')
DifferentiableLogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 7,
        'breadth': 3,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

# NLM critic parameters
nlmcritic_group = parser.add_argument_group('Neural Logic Machines critic')
LogicMachine.make_nlm_parser(
    nlmcritic_group, {
        'depth': 4,
        'breadth': 2,
        'residual': True,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlmcrit')
nlmcritic_group.add_argument(
    '--nlm-attributes-critic',
    type=int,
    default=2,
    metavar='N',
    help='number of output attributes in each group of each layer of the critic LogicMachine'
)

# MemNN parameters, works when model is 'memnet'.
memnet_group = parser.add_argument_group('Memory Networks')
MemoryNet.make_memnet_parser(memnet_group, {}, prefix='memnet')

parser.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')

method_group = parser.add_argument_group('Method')
method_group.add_argument(
    '--concat-worlds',
    type='bool',
    default=True,
    metavar='B',
    help='concat the features of objects of same id among two worlds accordingly'
)
method_group.add_argument(
    '--pred-depth',
    type=int,
    default=None,
    metavar='N',
    help='the depth of nlm used for prediction task')
method_group.add_argument(
    '--pred-weight',
    type=float,
    default=0.1,
    metavar='F',
    help='the linear scaling factor for prediction task')


data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--gen-method',
    default='dnc',
    choices=['dnc', 'edge'],
    help='method use to generate random graph')
data_gen_group.add_argument(
    '--gen-graph-pmin',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-pmax',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-max-len',
    type=int,
    default=4,
    metavar='N',
    help='maximum length of shortest path during training')
data_gen_group.add_argument(
    '--gen-test-len',
    type=int,
    default=4,
    metavar='N',
    help='length of shortest path during testing')
data_gen_group.add_argument(
    '--gen-directed', action='store_true', help='directed graph')


MiningTrainerBase.make_trainer_parser(
    parser, {
        'epochs': 1000,
        'epoch_size': 100,
        'test_epoch_size': 50,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
        'curriculum_start': 2,
        'curriculum_step': 1,
        'curriculum_graduate': 12,
        'curriculum_thresh_relax': 0.01,
        'curriculum_thresh': 1,
        #don't learn well on several lessons at a time with PPO/multiple trajectories
        'sample_array_capacity': 1,
        'disable-balanced-sample': True,
        'inherit_neg_data': False,
        'enable_mining': True,
        'mining_interval': 10, #6 for sort/path
        'mining_epoch_size': 200,
        'mining_dataset_size': 200,
        'prob_pos_data': 0.6 #0.5 for sort/path
    })

train_group = parser.add_argument_group('Train')
train_group.add_argument('--seed', type=int, default=None, metavar='SEED')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=0.9,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient (default: 1)')
train_group.add_argument(
    '--ntrajectory',
    type=int,
    default=5,
    metavar='N',
    help='number of trajectories to compute gradient')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for extra prediction')
train_group.add_argument(
    '--candidate-relax',
    type=int,
    default=0,
    metavar='N',
    help='number of thresh relaxation for candidate')
train_group.add_argument(
    '--extract-path', action='store_true', help='extract path or not')
train_group.add_argument(
    '--extract-rule', action='store_true', help='extract rule')
train_group.add_argument(
    '--incr-llonly',
    type='bool',
    default=False)
train_group.add_argument(
    '--incr-rmunused',
    type='bool',
    default=False)
train_group.add_argument(
    '--incr-rmunused-deg',
    type=int,
    default=1)
train_group.add_argument(
    '--incr-with-tail',
    type='bool',
    default=True)
train_group.add_argument(
    '--incr-with-even',
    type='bool',
    default=False)
train_group.add_argument(
    '--incr-with-distill',
    type='bool',
    default=False)
train_group.add_argument(
    '--gumbel-noise-begin',
    type=float,
    default=0.1)
train_group.add_argument(
    '--dropout-prob-begin',
    type=float,
    default=0.001)
train_group.add_argument(
    '--tau-begin',
    type=float,
    default=1)
train_group.add_argument(
    '--last-tau',
    type=float,
    default=0.01)
train_group.add_argument(
    '--entropy-reg',
    type=float,
    default=0.0,
    metavar='F',
    help='entropy regularization weight for interpretability')

rl_group = parser.add_argument_group('Reinforcement Learning')
rl_group.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='F',
    help='discount factor for accumulated reward function in reinforcement learning'
)
rl_group.add_argument(
    '--penalty',
    type=float,
    default=-0.01,
    metavar='F',
    help='a small penalty each step')
rl_group.add_argument(
    '--entropy-beta',
    type=float,
    default=0.0,
    metavar='F',
    help='entropy loss scaling factor')
rl_group.add_argument(
    '--entropy-beta-decay',
    type=float,
    default=0.8,
    metavar='F',
    help='entropy beta exponential decay factor')
rl_group.add_argument(
    '--critic-type',
    type=int,
    default=5)
rl_group.add_argument(
    '--noptepochs',
    type=int,
    default=2
)
rl_group.add_argument(
    '--epsilon',
    type=float,
    default=0.2
)
rl_group.add_argument(
    '--lam',
    type=float,
    default=0.9
)
rl_group.add_argument(
    '--clip-vf',
    type=float,
    default=0.2
)
rl_group.add_argument(
    '--no-shuffle-minibatch',
    type='bool',
    default=True)
rl_group.add_argument(
    '--no-adv-norm',
    type='bool',
    default=False)
rl_group.add_argument(
    '--dlm-noise',
    type=int,
    default=2,
    metavar='N',
    help='dlm noise handling')
rl_group.add_argument(
    '--distribution',
    type=int,
    default=1, #0 NLRL, 1 softmax, 2 move e^F
    metavar='N',
    help='distribution used to transform reasonning to action selection')
rl_group.add_argument(
    '--no-decay',
    type='bool',
    default=False)

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--dump-play',
    action='store_true',
    help='dump the trajectory of the plays for visualization')
io_group.add_argument(
    '--dump-fail-only', action='store_true', help='dump failure cases only')
io_group.add_argument(
    '--load-checkpoint',
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--early-drop-epochs',
    type=int,
    default=2000,
    metavar='N',
    help='epochs could spend for each lesson, early drop')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')
schedule_group.add_argument(
    '--test-not-graduated',
    action='store_true',
    help='test not graduated models also')

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.dump_play = args.dump_play and (args.dump_dir is not None)
args.epoch_size = args.epoch_size // args.ntrajectory

if args.dump_dir is not None:
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)
else:
    args.checkpoints_dir = None
    args.summary_file = None

if args.seed is not None:
    random.reset_global_seed(args.seed)

if 'nlrl' in args.task:
    args.concat_worlds = False
    args.penalty = None
    #TODO: dataset for pred_valid cannot contain unary+binary states
    args.pred_weight = 0.0

    # no curriculum learn for NLRL tasks
    args.curriculum_start = 4
    args.curriculum_graduate = 4
    args.mining_epoch_size = 20

    #not used
    args.test_number_begin = 4
    args.test_number_step = 1
    args.test_number_end = 4
elif args.task in ['sort', 'path']:
    args.concat_worlds = False
    args.pred_weight = 0.0
    if args.curriculum_start < 3:
        args.curriculum_start = 3

if args.task == 'sort':
    from difflogic.envs.algorithmic import make as make_env
elif args.task == 'path':
    from difflogic.envs.graph import make as make_env
    make_env = functools.partial(
      make_env,
      pmin=args.gen_graph_pmin,
      pmax=args.gen_graph_pmax,
      directed=args.gen_directed,
      gen_method=args.gen_method)
else:
    from difflogic.envs.blocksworld import make as make_env
    make_env = functools.partial(make_env, random_order=True, exclude_self=True, fix_ground=True)

logger = get_logger(__file__)


class Model(nn.Module):
    """The model for blocks world tasks."""

    def __init__(self):
        super().__init__()
        self.transform = InputTransform('cmp', exclude_self=False)

        input_dims = None
        # The 4 dimensions are: world_id, block_id, coord_x, coord_y
        if args.task == 'final':
            input_dim = 4
            # current_dim = 4 * 3 = 12
            current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
            self.feature_axis = 1 if args.concat_worlds else 2
        elif args.task == 'stack':
            input_dim = 2
            current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
            self.feature_axis = 2
        elif args.task == 'sort':
            self.feature_axis = 2
            current_dim = transformed_dim = 6
        elif args.task == 'path':
            self.feature_axis = 1
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[1] = 2
            input_dims[2] = 2
            transformed_dim = [0, 2, 2]
        elif 'nlrl' in args.task:
            self.feature_axis = 2
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[1] = 2 # unary: isFloor & top
            if args.task == 'nlrl-On':
                input_dims[2] = 2 # binary: goal_on & on
                transformed_dim = [0, 2, 2]
            else:
                input_dims[2] = 1 # binary: on
                transformed_dim = [0, 2, 1]
        else:
            raise ()

        if args.model in ['dlm', 'nlm'] and input_dims is None:
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[2] = current_dim

        if args.model == 'memnet':
            self.feature = MemoryNet.from_args(current_dim, self.feature_axis, args, prefix='memnet')
            current_dim = self.feature.get_output_dim()
        elif args.model == 'nlm':
            self.features = LogicMachine.from_args(input_dims, args.nlm_attributes, args, prefix='nlm')
            current_dim = self.features.output_dims[self.feature_axis]
        elif args.model == 'dlm':
            self.features = DifferentiableLogicMachine.from_args(input_dims, args.nlm_attributes, args, prefix='nlm')
            current_dim = self.features.output_dims[self.feature_axis]
        else:
            raise ()

        self.final_transform = InputTransform('concat', exclude_self=False)
        if args.task == 'final':
            if args.concat_worlds:
                current_dim = (self.final_transform.get_output_dim(current_dim) + transformed_dim) * 2

        if args.model == 'dlm':
            self.pred_valid = DLMInferenceBase(current_dim, 1, False, 'root_valid')
            self.pred = DLMInferenceBase(current_dim, 1, False, 'root')
            if args.distribution == 2:
                self.ac_selector = ActionSelector(current_dim)

            self.tau = args.tau_begin
            self.dropout_prob = args.dropout_prob_begin
            self.gumbel_prob = args.gumbel_noise_begin

            self.update_stoch()
            if args.entropy_reg != 0.0:
                self.lowernoise()
                self.restorenoise()
        else:
            self.pred_valid = LogicInference(current_dim, 1, [])
            self.pred = LogitsInference(current_dim, 1, [])
        self.loss = PPOLoss()
        self.pred_loss = nn.BCELoss()
        self.force_decay = False
        self.left_distill_epoch = -2
        self.distill = False
        self.working_policy = None
        self.train_succ_avg = None

        range_dims=[]
        for i in range(args.curriculum_start, args.curriculum_graduate + args.curriculum_step, args.curriculum_step):
            if args.task == "final":
                range_dims.append([(i+1)*2, (i+1)*2, transformed_dim])
            elif args.task == "stack":
                range_dims.append([(i+1), (i+1), transformed_dim])

        if args.task == "final":
            n_action=lambda x: (x[0]//2)*((x[0]//2)-1)
        elif args.task == "stack":
            n_action=lambda x: (x[0])*(x[0]-1)

        self.isQnet = True
        if args.critic_type == 0:
            self.critic = InvariantNObject(MLPCritic, range_dims, dict())
            self.isQnet = False
        elif args.critic_type == 1:
            self.critic = GRUCritic(transformed_dim)
            self.isQnet = False
        elif args.critic_type == 2:
            self.critic = GRUCritic(transformed_dim, shuffle_index=True)
            self.isQnet = False
        elif args.critic_type == 3:
            nlmcrit = LogicMachine.from_args(input_dims, args.nlm_attributes_critic, args, prefix='nlmcrit')
            self.critic = InvariantNObject(NLMMLPCritic, range_dims, dict(nlm=nlmcrit, feature_axis=None if args.model != 'dlm' else 0))
            self.isQnet = False
        elif args.critic_type == 4:
            self.critic = InvariantNObject(ConvCritic, range_dims, dict(input_channel=transformed_dim))
            self.isQnet = False
        elif args.critic_type == 5:
            self.critic = MixedGRUCritic(transformed_dim)
            self.isQnet = False
        elif args.critic_type == 6:
            self.critic = InvariantNObject(MLPCriticQ, range_dims, dict(n_action_func=n_action))
        elif args.critic_type == 7:
            nlmcrit = LogicMachine.from_args(input_dims, args.nlm_attributes_critic, args, prefix='nlmcrit')
            self.critic = InvariantNObject(NLMMLPCriticQ, range_dims, dict(nlm=nlmcrit, n_action_func=n_action, feature_axis=None if args.model != 'dlm' else 0))
        elif args.critic_type == 8:
            self.critic = InvariantNObject(ConvReduceCritic, range_dims, dict())
            self.isQnet = False
        elif args.critic_type == 9:
            self.critic = InvariantNObject(ConvReduceCriticQ, range_dims, dict(n_action_func=n_action))
        else:
            print("unkown critic_type")
            quit()
        self.critic_loss = nn.MSELoss()


    def update_stoch(self):
        if args.model == 'dlm':
            self.features.update_tau(self.tau)
            self.pred_valid.update_tau(self.tau)
            self.pred.update_tau(self.tau)

            self.features.update_gumbel_noise(self.gumbel_prob)
            self.pred_valid.update_gumbel_noise(self.gumbel_prob)
            self.pred.update_gumbel_noise(self.gumbel_prob)

            self.features.update_dropout_prob(self.dropout_prob)
            self.pred_valid.update_dropout_prob(self.dropout_prob)
            self.pred.update_dropout_prob(self.dropout_prob)

    def lowernoise(self):
        if args.model == 'dlm':
            self.pred.independant_noise_per_sample = False
            self.pred.with_gumbel = False
            self.pred.with_dropout = False

            self.pred_valid.independant_noise_per_sample = False
            self.pred_valid.with_gumbel = False
            self.pred_valid.with_dropout = False

            self.features.independant_noise_per_sample(False)
            self.features.with_gumbel(False)
            self.features.with_dropout(False)

    def restorenoise(self):
        if args.model == 'dlm':
            self.pred.independant_noise_per_sample = True
            self.pred_valid.independant_noise_per_sample = True
            self.features.independant_noise_per_sample(True)

            self.pred.with_gumbel = True
            self.pred_valid.with_gumbel = True
            self.features.with_gumbel(True)

            if args.entropy_reg == 0.0:
                self.pred.with_dropout = True
                self.pred_valid.with_dropout = True
                self.features.with_dropout(True)

    def stoch_decay(self, lesson, train_succ):
        reload_optimizer = False

        if (args.model == 'dlm' and not args.no_decay and train_succ > 0.95 and not self.distill) or self.force_decay:
            if not self.force_decay:
                self.train_succ_avg = collections.deque(maxlen=10)
                if args.incr_with_distill:
                    logger.info("Save policy")
                    self.working_policy = copy.deepcopy(self)
                    self.working_policy.features.continuous_relaxation()
                    self.working_policy.distill = False
                    self.working_policy.lowernoise()
            if self.train_succ_avg:
                self.train_succ_avg.append(train_succ)
            self.force_decay = True
            self.tau = self.tau * 0.98
            self.gumbel_prob = self.gumbel_prob * 0.98
            self.dropout_prob = self.dropout_prob * 0.98
            args.pred_weight = args.pred_weight * 0.98

            #considered it failed
            if self.tau <= 0.005 or \
                    (self.train_succ_avg and len(self.train_succ_avg) >= self.train_succ_avg.maxlen and
                     (sum(self.train_succ_avg) / len(self.train_succ_avg)) < 0.85):
                logger.info("Failed to extract interpretable policy ")
                if self.tau <= 0.005:
                    logger.info("because tau is too low")
                else:
                    logger.info("because succ rate too impacted")
                self.tau = args.tau_begin
                self.dropout_prob = args.dropout_prob_begin
                self.gumbel_prob = args.gumbel_noise_begin
                self.features.reset_weights()
                self.pred.reset_weights()
                reload_optimizer = True
                if args.incr_with_distill:
                    self.left_distill_epoch = 30
                # else:
                #     self.force_decay = False

            self.update_stoch()

        return reload_optimizer

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)

        if args.task in ['final', 'stack', 'sort']:
            states = feed_dict.states.float()
            batch_size = states.size(0)
        else:
            states = feed_dict.states
            batch_size = states[0].size(0)

        f, fs, more_info = self.get_binary_relations(states)
        saved_for_fa = f
        if args.model == 'dlm':
            f = self.pred(f)
            logits = f[0].squeeze(dim=-1).view(batch_size, -1)
            logits = 1e-5 + logits * (1.0 - 2e-5)
            if args.distribution == 0:
                sigma = logits.sum(-1).unsqueeze(-1)
                policy = torch.where(sigma > 1.0, logits/sigma, logits + (1-sigma)/logits.shape[1])
            elif args.distribution == 1:
                policy = F.softmax(logits / args.last_tau, dim=-1).clamp(min=1e-20)
            elif args.distribution == 2:
                if self.training:
                    fa = self.ac_selector(saved_for_fa.detach())
                    policy = (fa.sigmoid() + 1e-5 )*logits
                else:
                    policy = logits
                policy = policy / policy.sum(-1).unsqueeze(-1)
            else:
                raise()

            if feed_dict.training:
                if 'saturation' in more_info.keys():
                    more_info['saturation'].extend(f[1]['saturation'])
                else:
                    more_info['saturation'] = [f[1]['saturation']]
                if 'entropies' in more_info.keys():
                    more_info['entropies'].extend(f[1]['entropies'])
        else:
            logits = self.pred(f).squeeze(dim=-1).view(batch_size, -1)
            policy = F.softmax(logits, dim=-1).clamp(min=1e-20)

        if feed_dict.eval_critic:
            if self.isQnet:
                crit_out = self.critic(fs)
                approxr = (crit_out*policy).sum(-1)
            else:
                approxr = crit_out = self.critic(fs)
        else:
            crit_out = None

        if not feed_dict.training:
            if self.distill and self.working_policy is None and 'previous_tail' in more_info.keys():
                logits_previous_tail = more_info['previous_tail'].squeeze(dim=-1).view(batch_size, -1)
                if args.distribution == 0:
                    sigma = logits_previous_tail.sum(-1).unsqueeze(-1)
                    policy_previous_tail = torch.where(sigma > 1.0, logits_previous_tail / sigma, logits_previous_tail + (1 - sigma) / logits_previous_tail.shape[1])
                elif args.distribution == 1:
                    policy_previous_tail = F.softmax(logits_previous_tail / args.last_tau, dim=-1).clamp(min=1e-20)
                elif args.distribution == 2:
                    if self.training:
                        fa = self.ac_selector(saved_for_fa.detach())
                        policy_previous_tail = (fa.sigmoid() + 1e-5) * logits_previous_tail
                    else:
                        policy_previous_tail = logits_previous_tail
                    policy_previous_tail = policy_previous_tail / policy_previous_tail.sum(-1).unsqueeze(-1)
                else:
                    raise()
                return dict(policy=policy_previous_tail, logits=logits_previous_tail, value=crit_out)
            elif self.distill:
                feed_dict_out = self.working_policy(feed_dict)
                return dict(policy=feed_dict_out['policy'], logits=feed_dict_out['logits'], value=feed_dict_out['value'])
            return dict(policy=policy, logits=logits, value=crit_out)

        if self.distill and self.working_policy is None:
            loss = self.pred_loss(policy, more_info['previous_tail'].squeeze(dim=-1).view(batch_size, -1))
            monitors = dict()
            monitors['distill_loss'] = loss
        elif self.distill:
            with torch.set_grad_enabled(False):
                f_teacher, _, _ = self.working_policy.get_binary_relations(states)
                f_teacher = self.working_policy.pred(f_teacher)[0].squeeze(dim=-1).view(batch_size, -1)
            loss = self.pred_loss(logits, f_teacher)
            monitors = dict()
            monitors['distill_loss'] = loss
        else:
            loss, monitors = self.loss(policy, feed_dict.old_policy, feed_dict.actions, feed_dict.advantages, args.epsilon, feed_dict.entropy_beta)

        if self.isQnet:
            crit_preloss = (crit_out * feed_dict.actions_ohe).sum(-1)
        else:
            crit_preloss = crit_out
        if args.clip_vf is not None and args.clip_vf > 0.0:
            if self.isQnet:
                previous_val = (feed_dict.values*feed_dict.actions_ohe).sum(-1)
                crit_preloss = previous_val + torch.clamp(crit_preloss - previous_val, -args.clip_vf, args.clip_vf)
            else:
                crit_preloss = feed_dict.values + torch.clamp(crit_preloss - feed_dict.values, -args.clip_vf, args.clip_vf)
        losscrit = self.critic_loss(crit_preloss, feed_dict.returns)
        monitors['critic_accuracy'] = losscrit
        loss += losscrit
        if args.pred_weight != 0.0:
            pred_states = feed_dict.pred_states.float()
            f, _, _ = self.get_binary_relations(pred_states, depth=args.pred_depth)
            if args.model == 'dlm':
                f = self.pred_valid(f)[0].squeeze(dim=-1).view(pred_states.size(0), -1)
            else:
                f = self.pred_valid(f).squeeze(dim=-1).view(pred_states.size(0), -1)
            # Set minimal value to avoid loss to be nan.
            valid = f[range(pred_states.size(0)), feed_dict.pred_actions].clamp(min=1e-20)
            pred_loss = self.pred_loss(valid, feed_dict.valid)
            monitors['pred/accuracy'] = feed_dict.valid.eq((valid > 0.5).float()).float().mean()
            loss = loss + args.pred_weight * pred_loss

        if args.model == 'dlm':
            pred = (logits.detach().cpu() > 0.5).float()
            sat = 1 - (logits.detach().cpu() - pred).abs()
            monitors.update({'saturation/min': np.array(sat.min())})
            monitors.update({'saturation/mean': np.array(sat.mean())})
            saturation_inside = torch.cat([a.flatten() for a in more_info['saturation']])
            monitors.update({'saturation-inside/min': np.array(saturation_inside.cpu().min())})
            monitors.update({'saturation-inside/mean': np.array(saturation_inside.cpu().mean())})
            monitors.update({'tau': np.array(self.tau)})
            monitors.update({'dropout_prob': np.array(self.dropout_prob)})
            monitors.update({'gumbel_prob': np.array(self.gumbel_prob)})

            if args.entropy_reg != 0.0:
                entropies = torch.cat([a.flatten() for a in more_info['entropies']])
                loss += args.entropy_reg*entropies.mean()

        return loss, monitors, dict()

    def get_binary_relations(self, states, depth=None):
        """get binary relations given states, up to certain depth."""
        more_info = None
        if args.task in ['final', 'stack']:
            # total = 2 * the number of objects in each world
            total = states.size()[1]
            f = self.transform(states)
            fs = [None, f]
        elif args.task == 'sort':
            f = states
            fs = [None, f]
        else:
            f = states
            fs = f

        if args.model == 'memnet':
            f = self.feature(f)
        else:
            inp = [None for i in range(args.nlm_breadth + 1)]
            if type(f) is not list:
                inp[2] = f
            else:
                inp[1] = f[0]
                inp[2] = f[1]
            if args.model == 'dlm':
                if not self.training and args.extract_path:
                    self.features.extract_graph(self.feature_axis, self.pred)
                    for i in range(len(inp)):
                        if inp[i] is None:
                            continue
                        inp[i] = inp[i].bool()
                features = self.features(inp, depth=depth, extract_rule=args.extract_rule)
                f = features[0][self.feature_axis]
                more_info = features[1]
            else:
                features = self.features(inp, depth=depth)
                f = features[self.feature_axis]

        if args.task == 'final':
            assert total % 2 == 0
            nr_objects = total // 2
            if args.concat_worlds:
                # To concat the properties of blocks with the same id in both world.
                f = torch.cat([f[:, :nr_objects], f[:, nr_objects:]], dim=-1)
                states = torch.cat([states[:, :nr_objects], states[:, nr_objects:]], dim=-1)
                transformed_input = self.transform(states)
                # And perform a 'concat' transform to binary representation (relations).
                f = torch.cat([self.final_transform(f), transformed_input], dim=-1)
            else:
                f = f[:, :nr_objects, :nr_objects].contiguous()
        elif args.task == 'stack' or 'nlrl' in args.task:
            nr_objects = total if args.task == 'stack' else states[0].size()[1]
            f = f[:, :nr_objects, :nr_objects].contiguous()
        elif args.task in ['sort', 'path']:
            pass
        else:
            raise ()

        if args.task != 'path':
            f = meshgrid_exclude_self(f)
        return f, fs, more_info


def make_data(traj, gamma, succ, last_next_value, lam, isQnet):
    """Aggregate data as a batch for RL optimization."""


    traj['actions'] = as_tensor(np.array(traj['actions']))
    traj['actions_ohe'] = as_tensor(np.array(traj['actions_ohe']))
    if type(traj['states'][0]) is list:
        f1 = [f[0] for f in traj['states']]
        f2 = [f[1] for f in traj['states']]
        traj['states'] = [torch.cat(f1, dim=0), torch.cat(f2, dim=0)]
    else:
        traj['states'] = torch.cat(traj['states'], dim=0)
    traj['values'] = torch.cat(traj['values'], dim=0)
    traj['old_policy'] = torch.cat(traj['old_policy'], dim=0)
    traj['advantages'] = torch.zeros(traj['values'].shape[0])
    last_gae_lam = 0  # the next state of the last state in traj is always the terminal state.
    for step in reversed(range(len(traj['values']))):
        if step == len(traj['values']) - 1:
            next_values = 0 if succ else last_next_value
        elif isQnet:
            next_values = (traj['values'][step + 1] * traj['old_policy'][step + 1].cpu()).sum(-1)
        else:
            next_values = traj['values'][step + 1]
        if isQnet:
            delta = traj['rewards'][step] + gamma * next_values - (traj['values'][step]*traj['actions_ohe'][step]).sum(-1)
        else:
            delta = traj['rewards'][step] + gamma * next_values - traj['values'][step]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        traj['advantages'][step] = last_gae_lam

    if isQnet:
        traj['returns'] = traj['advantages'] + (traj['values'] * traj['actions_ohe']).sum(-1)
    else:
        traj['returns'] = traj['advantages'] + traj['values']

    return traj


def run_episode(env,
                model,
                mode,
                number,
                play_name='',
                dump=False,
                dataset=None,
                eval_only=False,
                use_argmax=False,
                need_restart=False,
                entropy_beta=0.0,
                eval_critic=False):
    """Run one episode using the model with $number blocks."""
    is_over = False
    traj = collections.defaultdict(list)
    score = 0
    if need_restart:
        env.restart()

    optimal = None
    if args.task == 'path':
        optimal = env.unwrapped.dist
        relation = env.unwrapped.graph.get_edges()
        relation = np.stack([relation, relation.T], axis=-1).astype(dtype=np.float32)
        st, ed = env.current_state
        nodes_trajectory = [int(st)]
        destination = int(ed)
        policies = []
    elif args.task == 'sort':
        optimal = env.unwrapped.optimal
        array = [str(i) for i in env.unwrapped.array]

    # If dump_play=True, store the states and actions in a json file
    # for visualization.
    dump_play = args.dump_play and dump
    if dump_play:
        nr_objects = number + 1
        array = env.unwrapped.current_state
        moves, new_pos, policies = [], [], []

    if args.model == 'dlm':
        # by default network isn't in training mode during data collection
        # but with dlm we don't want to use argmax only
        # except in 2 cases (testing the interpretability or the mining phase to get an interpretable policy):
        if ('inter' in mode) or ('mining' in mode) or ('inherit' in mode):
            model.lowernoise()
            model.features.extract_graph(model.feature_axis, model.pred, track_duplicates=('mining' in mode))
        else:
            model.train(True)
            model.features.continuous_relaxation()

            if args.dlm_noise == 1 and (('mining' in mode) or ('inherit' in mode) or ('test' in mode)):
                model.lowernoise()
            elif args.dlm_noise == 2:
                model.lowernoise()

    step = 0
    while not is_over:
        if args.task == 'path':
            st, ed = env.current_state
            state = np.zeros((relation.shape[0], 2), dtype=np.float32)
            state[st, 0] = 1
            state[ed, 1] = 1
            feed_dict = dict(states=[np.array([state]), np.array([relation])])
        else:
            state = env.current_state
            if 'nlrl' not in args.task or args.task == 'sort':
                feed_dict = dict(states=np.array([state]))
            else:
                feed_dict = dict(states=state)
        feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
        feed_dict['eval_critic'] = as_tensor(eval_critic)
        feed_dict['training'] = as_tensor(False)
        feed_dict = as_tensor(feed_dict)
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)

        with torch.set_grad_enabled(False):
            output_dict = model(feed_dict)
        policy = output_dict['policy']
        p = as_numpy(policy.data[0])
        action = p.argmax() if use_argmax else random.choice(len(p), p=p)
        if args.pred_weight != 0.0:
            # Need to ensure that the env.utils.MapActionProxy is the outermost class.
            mapped_x, mapped_y = env.mapping[action]
            # env.unwrapped to get the innermost Env class.
            valid = env.unwrapped.world.moveable(mapped_x, mapped_y)
        reward, is_over = env.action(action)
        step += 1
        if dump_play:
            moves.append([mapped_x, mapped_y])
            res = tuple(env.current_state[mapped_x][2:])
            new_pos.append((int(res[0]), int(res[1])))

            logits = as_numpy(output_dict['logits'].data[0])
            tops = np.argsort(p)[-10:][::-1]
            tops = list(
                map(lambda x: (env.mapping[x], float(p[x]), float(logits[x])), tops))
            policies.append(tops)
        # For now, assume reward=1 only when succeed, otherwise reward=0.
        # Manipulate the reward and get success information according to reward.
        if reward == 0 and args.penalty is not None:
            reward = args.penalty
        succ = 1 if is_over and reward > 0.99 else 0

        score += reward
        if not eval_only:
            traj['values'].append(output_dict['value'].detach().cpu())
            if type(feed_dict['states']) is list:
                traj['states'].append([f.detach().cpu() for f in feed_dict['states']])
            else:
                traj['states'].append(feed_dict['states'].detach().cpu())
            traj['rewards'].append(reward)
            traj['actions'].append(action)
            traj['actions_ohe'].append(to_categorical(action, num_classes=policy.shape[1]))
            traj['old_policy'].append(policy.detach().cpu())
        if args.pred_weight != 0.0:
            if not eval_only and dataset is not None and mapped_x != mapped_y:
                dataset.append(number + 1, state, action, valid)

    if eval_critic:
        if args.task == 'path':
            st, ed = env.current_state
            state = np.zeros((relation.shape[0], 2), dtype=np.float32)
            state[st, 0] = 1
            state[ed, 1] = 1
            feed_dict = dict(states=[np.array([state]), np.array([relation])])
        else:
            state = env.current_state
            if 'nlrl' not in args.task:
                feed_dict = dict(states=np.array([state]))
            else:
                feed_dict = dict(states=state)
        feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
        feed_dict['eval_critic'] = as_tensor(eval_critic)
        feed_dict['training'] = as_tensor(False)
        feed_dict = as_tensor(feed_dict)
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)

        with torch.set_grad_enabled(False):
            output_dict = model(feed_dict)
        if model.isQnet:
            last_next_value = (output_dict['value'].detach() * output_dict['policy'].detach()).sum(-1).cpu().numpy()
        else:
            last_next_value = output_dict['value'].detach().cpu().numpy()
    else:
        last_next_value = None

    # Dump json file as record of the playing.
    if dump_play and not (args.dump_fail_only and succ):
        array = array[:, 2:].astype('int32').tolist()
        array = [array[:nr_objects], array[nr_objects:]]
        json_str = json.dumps(
            # Let indent=True for an indented view of json files.
            dict(array=array, moves=moves, new_pos=new_pos,
                 policies=policies))
        dump_file = os.path.join(
            args.current_dump_dir,
            '{}_blocks{}.json'.format(play_name, env.unwrapped.nr_blocks))
        with open(dump_file, 'w') as f:
            f.write(json_str)

    length = step

    if args.model == 'dlm':
        model.restorenoise()

    return succ, score, traj, length, last_next_value, optimal


class MyTrainer(MiningTrainerBase):

    def __init__(self, model, optimizer, epochs, epoch_size, test_epoch_size, test_number_begin, test_number_step,
                 test_number_end, curriculum_start, curriculum_step, curriculum_graduate, enable_candidate, curriculum_thresh,
                 curriculum_thresh_relax, curriculum_force_upgrade_epochs, sample_array_capacity, enhance_epochs, enable_mining,
                 repeat_mining, candidate_mul, mining_interval, mining_epoch_size, mining_dataset_size, inherit_neg_data,
                 disable_balanced_sample, prob_pos_data):
        super().__init__(model, optimizer, epochs, epoch_size, test_epoch_size, test_number_begin, test_number_step, test_number_end,
                         curriculum_start, curriculum_step, curriculum_graduate, enable_candidate, curriculum_thresh,
                         curriculum_thresh_relax, curriculum_force_upgrade_epochs, sample_array_capacity, enhance_epochs, enable_mining,
                         repeat_mining, candidate_mul, mining_interval, mining_epoch_size, mining_dataset_size, inherit_neg_data,
                         disable_balanced_sample, prob_pos_data)
        self.should_extract = False

    def save_checkpoint(self, name):
        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir, 'checkpoint_{}.pth'.format(name))
            super().save_checkpoint(checkpoint_file, extra=self.model)

    def load_checkpoint(self, filename):
        del self._model
        self._model = torch.load(filename)['extra']

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['epoch'] = self.current_epoch
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    def _prepare_dataset(self, epoch_size, mode):
        pass

    def _get_player(self, number, mode, index):
        if args.task == 'path':
            test_len = args.gen_test_len
            test_mode = (args.model == 'dlm' and number == args.curriculum_graduate and "mining" in mode) or ("test" in mode)
            dist_range = (test_len, test_len) if test_mode else (1, args.gen_max_len)
            player = make_env(args.task, number, dist_range=dist_range)
        elif 'nlrl' not in args.task or 'test' not in mode:
            player = make_env(args.task, number)
        else:
            #testing env. for NLRL
            #suppose 5 variations per env.
            player = make_env(args.task, number, variation_index=(index%5))

        player.restart()
        return player

    def _get_result_given_player(self, index, meters, number, player, mode):
        assert mode in ['train', 'test', 'mining', 'mining-deter', 'mining-stoch', 'inherit', 'test-inter', 'test-inter-deter', 'test-deter']
        params = dict(
            eval_only=True,
            eval_critic=False,
            number=number,
            play_name='{}_epoch{}_episode{}'.format(mode, self.current_epoch, index))
        backup = None
        if mode == 'train':
            params['eval_only'] = False
            params['eval_critic'] = True
            params['dataset'] = self.valid_action_dataset
            params['entropy_beta'] = self.entropy_beta
            meters.update(lr=self.lr, entropy_beta=self.entropy_beta)
        elif 'test' in mode:
            params['dump'] = True
            params['use_argmax'] = 'deter' in mode
        else:
            backup = copy.deepcopy(player)
#            params['use_argmax'] = self.is_candidate
            params['use_argmax'] = 'deter' in mode

        if mode == 'train':
            mergedfc = []
            for i in range(args.ntrajectory):
                succ, score, traj, length, last_next_value, optimal = run_episode(player, self.model, mode, need_restart=(i!=0), **params)
                if args.task in ['sort', 'path']:
                    meters.update(number=number, succ=succ, score=score, length=length, optimal=optimal)
                else:
                    meters.update(number=number, succ=succ, score=score, length=length)
                feed_dict = make_data(traj, args.gamma, succ, last_next_value[0], lam=args.lam, isQnet=self.model.isQnet)
                # content from valid_move dataset
                if args.pred_weight != 0.0:
                    states, actions, labels = self.valid_action_dataset.sample_batch(args.batch_size)
                    feed_dict['pred_states'] = as_tensor(states)
                    feed_dict['pred_actions'] = as_tensor(actions)
                    feed_dict['valid'] = as_tensor(labels).float()
                mergedfc.append(feed_dict)

            for k in feed_dict.keys():
                if k not in ["rewards"]: #reward not used to update loss
                    if type(mergedfc[0][k]) is list:
                        f1 = [j[k][0] for j in mergedfc]
                        f2 = [j[k][1] for j in mergedfc]
                        feed_dict[k] = [torch.cat(f1, dim=0), torch.cat(f2, dim=0)]
                    else:
                        feed_dict[k] = torch.cat([j[k] for j in mergedfc], dim=0)
            feed_dict['entropy_beta'] = as_tensor(self.entropy_beta).float()
            feed_dict['eval_critic'] = as_tensor(True)
            feed_dict['training'] = as_tensor(True)

            if feed_dict['advantages'].shape[0] > 1 and not args.no_adv_norm:
                feed_dict['advantages'] = (feed_dict['advantages'] - feed_dict['advantages'].mean())/(feed_dict['advantages'].std() + 10**-7)

            if args.use_gpu:
                feed_dict = as_cuda(feed_dict)
            self.model.train()
            return feed_dict
        else:
            succ, score, traj, length, last_next_value, optimal = run_episode(player, self.model, mode, **params)
            if args.task in ['sort', 'path']:
                meters.update(number=number, succ=succ, score=score, length=length, optimal=optimal)
                message = ('> {} iter={iter}, number={number}, succ={succ}, '
                       'score={score:.4f}, length={length}, optimal={optimal}').format(mode, iter=index, **meters.val)
            else:
                meters.update(number=number, succ=succ, score=score, length=length)
                message = ('> {} iter={iter}, number={number}, succ={succ}, '
                       'score={score:.4f}, length={length}').format(mode, iter=index, **meters.val)
            return message, dict(succ=succ, number=number, backup=backup)

    def _extract_info(self, extra):
        return extra['succ'], extra['number'], extra['backup']

    def _get_accuracy(self, meters):
        return meters.avg['succ']

    def _get_threshold(self):
        candidate_relax = 0 if self.is_candidate else args.candidate_relax
        return super()._get_threshold() - \
               self.curriculum_thresh_relax * candidate_relax

    def _upgrade_lesson(self):
        super()._upgrade_lesson()
        # Adjust lr & entropy_beta w.r.t different lesson progressively.
        self.lr *= args.lr_decay
        self.entropy_beta *= args.entropy_beta_decay
        self.set_learning_rate(self.lr)

        if not (self.is_candidate or self.is_graduated):
            self.model.force_decay = True
            self.model.working_policy = None
            self.should_extract = True

    def _train_epoch(self, epoch_size):
        if self.should_extract:
            logger.info("TRIGGERED extraction for new lesson " + str(sum(p.numel() for p in self.model.features.parameters())) +
                        " parameters with " + str(np.array(self.model.features.input_dims).sum()) + " predicates")
            self.model.tau = args.tau_begin
            self.model.dropout_prob = args.dropout_prob_begin
            self.model.gumbel_prob = args.gumbel_noise_begin
            self.model.update_stoch()

            new_features, entries_size = self.model.features.extract_stacked(self.model.feature_axis, self.model.pred, onlyLastLayer=args.incr_llonly,
                                                                             removeOldUnused=args.incr_rmunused, with_tail=args.incr_with_tail,
                                                                             with_even=args.incr_with_even, removeOldUnusedDeg=args.incr_rmunused_deg,
                                                                             store_previous_tail=args.incr_with_distill)
            new_pred = DLMInferenceBase(new_features.output_dims[self.model.feature_axis], 1, False, 'root')
            new_pred.to(next(self.model.parameters()).device)
            self.model.pred = new_pred
            self.model.features = new_features
            self._optimizer = get_optimizer(args.optimizer, self.model, args.lr)
            logger.info("TRIGGERED AFTER " + str(sum(p.numel() for p in self.model.features.parameters())) +
                        " parameters with " + str(entries_size) + " predicates")
            if args.incr_with_distill:
                self.model.left_distill_epoch = 30
                self.model.force_decay = True

            self.should_extract = False

        model = self.model
        model.distill = (model.left_distill_epoch >= 0)
        if model.left_distill_epoch == -1:
            model.force_decay = False
            model.train_succ_avg = None
        model.left_distill_epoch -= 1
        meters = GroupMeters()
        self._prepare_dataset(epoch_size, mode='train')

        def train_func(index):
            model.eval()
            feed_dict = self._get_train_data(index, meters)
            model.train()
            if not args.no_shuffle_minibatch:
                nbatch = feed_dict['states'].shape[0]
                minibatch_size = args.batch_size
                inds = np.arange(nbatch)
                np.random.shuffle(inds)
                for _ in range(args.noptepochs):
                    for start in range(0, nbatch, minibatch_size):
                        end = start + minibatch_size
                        mbinds = inds[start:end]
                        subfeed_dict = {}
                        for k in feed_dict.keys():
                            if type(feed_dict[k]) == torch.Tensor and len(feed_dict[k].shape) != 0 and k not in ('pred_states', 'pred_actions', 'valid'):
                                subfeed_dict[k] = feed_dict[k][mbinds, ...]
                            else:
                                subfeed_dict[k] = feed_dict[k]
                        message, _ = self._train_step(subfeed_dict, meters)
            else:
                for _ in range(args.noptepochs):
                    message, _ = self._train_step(feed_dict, meters)
            return message

        # For $epoch_size times, do train_func with tqdm progress bar.
        tqdm_for(epoch_size, train_func)
        logger.info(
            meters.format_simple(
                '> Train Epoch {:5d}: '.format(self.current_epoch),
                compressed=False))
        self._dump_meters(meters, 'train')

        reload_optimizer = self.model.stoch_decay(self.current_number, meters.avg['succ'])
        if reload_optimizer:
            self._optimizer = get_optimizer(args.optimizer, self.model, self.lr)
        elif not self.is_graduated and not self.model.distill:
            self._take_exam(train_meters=copy.copy(meters))

        i = self.current_epoch
        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        if args.test_interval is not None and i % args.test_interval == 0:
            self.test()

        return meters

    def _early_stop(self, meters):
        t = args.early_drop_epochs
        if t is not None and self.current_epoch > t * (self.nr_upgrades + 1):
            return True
        return super()._early_stop(meters)

    def train(self):
        self.valid_action_dataset = ValidActionDataset()
        self.lr = args.lr
        self.entropy_beta = args.entropy_beta
        return super().train()

    def test(self):
        ret1 = super().test()
        ret2 = super().advanced_test(inter=False, deter=True)

        if args.model != 'dlm':
            return ret1 if ret1[-1].avg['score'] > ret2[-1].avg['score'] else ret2

        ret1 = super().advanced_test(inter=True, deter=False)
        ret2 = super().advanced_test(inter=True, deter=True)
        return ret1 if ret1[-1].avg['score'] > ret2[-1].avg['score'] else ret2


def main(run_id):
    if args.dump_dir is not None:
        if args.runs > 1:
            args.current_dump_dir = os.path.join(args.dump_dir,
                                                 'run_{}'.format(run_id))
            io.mkdir(args.current_dump_dir)
        else:
            args.current_dump_dir = args.dump_dir
        args.checkpoints_dir = os.path.join(args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)
        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')

    logger.info(format_args(args))

    model = Model()
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    if args.accum_grad > 1:
        optimizer = AccumGrad(optimizer, args.accum_grad)

    if args.use_gpu:
        model.cuda()
    trainer = MyTrainer.from_args(model, optimizer, args)

    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.load_checkpoint)

    if args.test_only:
        trainer.current_epoch = 0
        return None, trainer.test()

    graduated = trainer.train()
    trainer.save_checkpoint('last')
    test_meters = trainer.test() if graduated or args.test_not_graduated else None
    return graduated, test_meters


if __name__ == '__main__':
    stats = []
    nr_graduated = 0

    for i in range(args.runs):
        graduated, test_meters = main(i)
        logger.info('run {}'.format(i + 1))

        if test_meters is not None:
            for j, meters in enumerate(test_meters):
                if len(stats) <= j:
                    stats.append(GroupMeters())
                stats[j].update(number=meters.avg['number'], test_succ=meters.avg['succ'])

            for meters in stats:
                logger.info('number {}, test_succ {}'.format(meters.avg['number'], meters.avg['test_succ']))

        if not args.test_only:
            nr_graduated += int(graduated)
            logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
            if graduated:
                for j, meters in enumerate(test_meters):
                    stats[j].update(grad_test_succ=meters.avg['succ'])
            if nr_graduated > 0:
                for meters in stats:
                    logger.info('number {}, grad_test_succ {}'.format(meters.avg['number'], meters.avg['grad_test_succ']))
