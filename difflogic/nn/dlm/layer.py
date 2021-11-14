import copy

import torch
import torch.nn as nn

from jacinle.logging import get_logger

from difflogic.nn.neural_logic.modules.dimension import Expander, Reducer, Permutation
from difflogic.nn.neural_logic.modules._utils import exclude_mask, mask_value
from difflogic.nn.dlm.neural_logic import DLMInferenceBase
import itertools
import numpy as np

__all__ = ['DLMLayer', 'DifferentiableLogicMachine']

from ...myutils import update_dict_list

logger = get_logger(__file__)


def _get_tuple_n(x, n, tp):
    """Get a length-n list of type tp."""
    assert tp is not list
    if isinstance(x, tp):
        x = [x, ] * n
    assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(tp)
    for i in x:
        assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
    return x

def merge(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return torch.cat([x, y], dim=-1)

class DLMLayer(nn.Module):
    """Logic Layers do one-step differentiable logic deduction.

  The predicates grouped by their number of variables. The inter group deduction
  is done by expansion/reduction, the intra group deduction is done by logic
  model.

  Args:
    breadth: The breadth of the logic layer.
    input_dims: the number of input channels of each input group, should consist
                with the inputs. use dims=0 and input=None to indicate no input
                of that group.
    output_dims: the number of output channels of each group, could
                 use a single value.
    exclude_self: Not allow multiple occurrence of same variable when
                  being True.
    residual: Use residual connections when being True.
  """

    def __init__(
            self,
            breadth,
            input_dims,
            output_dims,
            depth,
            exclude_self=True,
            residual=False,
            identifier='l0',
            dlm_intern_params=None,
    ):
        super().__init__()
        assert breadth > 0, 'Does not support breadth <= 0.'
        if breadth > 3:
            logger.warn(
                'Using DLMLayer with breadth > 3 may cause speed and memory issue.')

        self.max_order = breadth
        self.residual = residual
        self.depth = depth
        input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)

        output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

        self.logic, self.dim_perms, self.dim_expanders, self.dim_reducers = [
            nn.ModuleList() for _ in range(4)
        ]

        for i in range(self.max_order + 1):
            # collect current_dim from group i-1, i and i+1.
            current_dim = input_dims[i]
            if i > 0:
                expander = Expander(i - 1)
                self.dim_expanders.append(expander)
                current_dim += expander.get_output_dim(input_dims[i - 1])
            else:
                self.dim_expanders.append(None)

            if i + 1 < self.max_order + 1:
                reducer = Reducer(i + 1, exclude_self)
                self.dim_reducers.append(reducer)
                current_dim += reducer.get_output_dim(input_dims[i + 1])
            else:
                self.dim_reducers.append(None)

            if current_dim == 0:
                self.dim_perms.append(None)
                self.logic.append(None)
                output_dims[i] = 0
            else:
                perm = Permutation(i)
                self.dim_perms.append(perm)
                current_dim = perm.get_output_dim(current_dim)
                self.logic.append(
                    DLMInferenceBase(current_dim, output_dims[i], self.depth == 0, identifier + '_' + str(i), **dlm_intern_params))

        self.input_dims = input_dims
        self.output_dims = output_dims

        if self.residual:
            for i in range(len(input_dims)):
                self.output_dims[i] += input_dims[i]

        self.savePermutation = []
        for i in range(self.max_order + 1):
            self.savePermutation.append(list(itertools.permutations(range(1, i + 1))))

        self.isExpanded = [False for i in range(self.max_order + 1)]
        self.isInput = [False for i in range(self.max_order + 1)]
        self.isReduced = [False for i in range(self.max_order + 1)]

    def fakeForward(self, inputsSize, enable_feature_axis, feature_axis):  # return a fake output and the sizes I need to extract path between layers
        assert len(inputsSize) == self.max_order + 1
        paths = []
        outputsSize = []
        isResidual = [False for i in range(self.max_order + 1)]
        isExpanded = [False for i in range(self.max_order + 1)]
        isInput = [False for i in range(self.max_order + 1)]
        isReduced = [False for i in range(self.max_order + 1)]
        sizeResidual = [0 for i in range(self.max_order + 1)]
        sizeExpand = [0 for i in range(self.max_order + 1)]
        sizeInput = [0 for i in range(self.max_order + 1)]
        sizeReduce = [0 for i in range(self.max_order + 1)]
        if enable_feature_axis:
            for i in range(self.max_order + 1):
                if self.logic[i] is None:
                    outputsSize.append(None)
                    paths.append(None)
                else:
                    paths.append(self.logic[i].weight.argmax(-1))
                    pathSize = list(self.logic[i].weight.size()[:-1])
                    if i==feature_axis:
                        if i > 0 and self.input_dims[i - 1] > 0:
                            isExpanded[i] = True
                            sizeExpand[i] = self.input_dims[i - 1]
                        if i < len(inputsSize) and self.input_dims[i] > 0:
                            isInput[i] = True
                            sizeInput[i] = self.input_dims[i]
                        if i + 1 < len(inputsSize) and self.input_dims[i + 1] > 0:
                            isReduced[i] = True
                            sizeReduce[i] = self.input_dims[i + 1] * 2
                        if isExpanded[i]:
                            n = inputsSize[i][1] if i == 1 else None
                            if n is None:
                                n = inputsSize[i - 1][-2]
                            size = inputsSize[i - 1][:-1] + [n, ]  # +[(sizeExpand[i]+sizeInput[i]+sizeReduce[i])*2*np.math.factorial(i),]
                        elif isInput[i]:
                            size = inputsSize[i][:-1]  # +[(sizeExpand[i]+sizeInput[i]+sizeReduce[i])*2*np.math.factorial(i),]
                        else:
                            size = inputsSize[i + 1][:-2]  # +[(sizeExpand[i]+sizeInput[i]+sizeReduce[i])*2*np.math.factorial(i),]
                        outputSize = pathSize[1:-1] + size + pathSize[0:1]
                        if self.residual and self.input_dims[i] > 0:
                            isResidual[i] = True
                            sizeResidual = self.input_dims[i]
                            outputSize[-1] += self.input_dims[i]
                        outputsSize.append(outputSize)
            return outputsSize, isResidual, isExpanded, isInput, isReduced, sizeResidual, sizeExpand, sizeInput, sizeReduce, paths
        for i in range(self.max_order + 1):
            if self.logic[i] is None:
                outputsSize.append(None)
                paths.append(None)
            else:
                paths.append(self.logic[i].weight.argmax(-1))
                pathSize = list(self.logic[i].weight.size()[:-1])
                # from the original nlm:
                if i > 0 and self.input_dims[i - 1] > 0:
                    isExpanded[i] = True
                    sizeExpand[i] = self.input_dims[i - 1]
                if i < len(inputsSize) and self.input_dims[i] > 0:
                    isInput[i] = True
                    sizeInput[i] = self.input_dims[i]
                if i + 1 < len(inputsSize) and self.input_dims[i + 1] > 0:
                    isReduced[i] = True
                    sizeReduce[i] = self.input_dims[i + 1] * 2
                if isExpanded[i]:
                    n = inputsSize[i][1] if i == 1 else None
                    if n is None:
                        n = inputsSize[i - 1][-2]
                    size = inputsSize[i - 1][:-1] + [n, ]  # +[(sizeExpand[i]+sizeInput[i]+sizeReduce[i])*2*np.math.factorial(i),]
                elif isInput[i]:
                    size = inputsSize[i][:-1]  # +[(sizeExpand[i]+sizeInput[i]+sizeReduce[i])*2*np.math.factorial(i),]
                else:
                    size = inputsSize[i + 1][:-2]  # +[(sizeExpand[i]+sizeInput[i]+sizeReduce[i])*2*np.math.factorial(i),]
                outputSize = pathSize[1:-1] + size + pathSize[0:1]
                if self.residual and self.input_dims[i] > 0:
                    isResidual[i] = True
                    sizeResidual = self.input_dims[i]
                    outputSize[-1] += self.input_dims[i]
                outputsSize.append(outputSize)
        return outputsSize, isResidual, isExpanded, isInput, isReduced, sizeResidual, sizeExpand, sizeInput, sizeReduce, paths

    def forward(self, inputs, paths, extract_rule=False):
        assert len(inputs) == self.max_order + 1
        outputs = []
        other_outputs = {}
        if paths is not None:
            for i in range(len(inputs)):
                if inputs[i] != None:
                    useBooleanRepresentation = inputs[i].dtype == torch.bool
                    input_device = inputs[i].device
                    break
            for i in range(self.max_order + 1):
                if extract_rule:
                    print("breadth", i)
                if self.logic[i] is None:
                    outputs.append(None)
                else:
                    isExpanded = False
                    isInput = False
                    isReduced = False
                    sizeExpand = 0
                    sizeInput = 0
                    sizeReduce = 0
                    # print(f.shape)#[4, 10, 8]#[batch_size,training instances,predicates]
                    # print(self.logic[i].layer[0].weight.argmax(-1).shape)#[8, 2][#output attributes,nb_predicate_activated] #value is 0-15 for example, second half is negation
                    # path=self.logic[i].layer[0].weight.argmax(-1)
                    path = paths[i]
                    # from the original nlm:
                    if i > 0 and self.input_dims[i - 1] > 0:
                        isExpanded = True
                        sizeExpand = self.input_dims[i - 1]
                    if i < len(inputs) and self.input_dims[i] > 0:
                        isInput = True
                        sizeInput = self.input_dims[i]
                    if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                        isReduced = True
                        sizeReduce = self.input_dims[i + 1] * 2
                    # prepare f for logic[i](permute(f)):
                    # f=torch.zeros([4,10,16]), but only f[:,:,path[x,y]] matters
                    if isExpanded:
                        n = inputs[i].size(1) if i == 1 else None
                        if n is None:
                            n = inputs[i - 1].size()[-2]
                        size = inputs[i - 1].size()[:-1] + (n,) + ((sizeExpand + sizeInput + sizeReduce) * np.math.factorial(i),)
                    elif isInput:
                        size = inputs[i].size()[:-1] + ((sizeExpand + sizeInput + sizeReduce) * np.math.factorial(i),)
                    else:
                        size = inputs[i + 1].size()[:-2] + ((sizeExpand + sizeInput + sizeReduce) * np.math.factorial(i),)
                    f = torch.zeros(size, device=input_device, dtype=torch.bool if useBooleanRepresentation else torch.float32, requires_grad=False)

                    pset = set(path.cpu().numpy().flatten())
                    pset.discard(-1)
                    # for the same i
                    # p=p-negation*(self.logic[i].layer[0].weight.shape[-1]//2)
                    # p=p%(sizeExpand+sizeInput+sizeReduce)
                    # inputs[i - 1][...,p] or inputs[i][...,p-sizeExpand] or inputs[i+1][...,[(p-(sizeExpand+sizeInput))//2]]
                    # only need those inputs
                    # previous layer, output[i][...,j] is useless for example.
                    # torch.cat([ffand,ffor]).permute(...,0)
                    # for output[i][...,j], comes from output[i][j,...]
                    # means from ffand or ffor, f[j,[0,1],...] is useless
                    # means in the previous layer, weight[j,:] is useless
                    # back propagation: current layer set p --> modify previous
                    if len(pset)>0:
                        if extract_rule:
                            print(pset)
                            print(path)
                    for p in pset:
                        if extract_rule:
                            print('predicate',p)
                        if p == size[-1]:
                            continue
                        index = p
                        #negation = 2 * p >= self.logic[i].weight.shape[-1]
                        #p -= negation * (self.logic[i].weight.shape[-1] // 2)
                        perm = self.savePermutation[i][p // (sizeExpand + sizeInput + sizeReduce)]
                        p = p % (sizeExpand + sizeInput + sizeReduce)
                        if p < sizeExpand:
                            assert isExpanded
                            n = inputs[i].size(1) if i == 1 else None
                            predicate = self.dim_expanders[i](inputs[i - 1][..., p], n)
                            if extract_rule:
                                print("Expand predicate",p,"for breadth",i-1)
                        elif p < sizeExpand + sizeInput:
                            assert isInput
                            predicate = inputs[i][..., p - sizeExpand]
                            if extract_rule:
                                print("Input predicate",p- sizeExpand,"for breadth",i)
                        else:
                            assert isReduced
                            p = p - (sizeExpand + sizeInput)
                            if p % 2 == 0:
                                # exists
                                mask = exclude_mask(inputs[i + 1][..., [p // 2]], cnt=i + 1, dim=-2 - i)
                                predicate = torch.max(mask_value(inputs[i + 1][..., [p // 2]], mask, 0.0), dim=-2)[0]
                                if extract_rule:
                                    print("Exists predicate",p//2,"for breadth",i+1)
                            else:
                                # forall
                                mask = exclude_mask(inputs[i + 1][..., [p // 2]], cnt=i + 1, dim=-2 - i)
                                predicate = torch.min(mask_value(inputs[i + 1][..., [p // 2]], mask, 1.0), dim=-2)[0]
                                if extract_rule:
                                    print("Forall predicate",p//2,"for breadth",i+1)
                            predicate = predicate.squeeze(-1)
                        #if negation:
                        #    f[..., index] = (1 - predicate).permute((0,) + perm)
                        #else:
                        f[..., index] = predicate.permute((0,) + perm)
                        if extract_rule:
                            print("permute",(0,) + perm)

                        # give f [4,10,16], path [8,2], calcullate self.logic[i]([f,1-f]) [8,2,4,10] -->
                    # output=torch.zeros(path.size()+f.size()[:-1])
                    # for x in range(path.size(0)):
                    #  for y in range(path.size(1)):
                    #     output[x,y,...]=f[...,path[x,y]]
                    f2 = torch.ones((*f.shape[:-1], 1), device=f.device, dtype=torch.bool if useBooleanRepresentation else torch.float32)
                    f = torch.cat((f, f2), -1)

                    output = f[..., path].permute(-2, -1, *range(len(f.shape) - 1))

                    if useBooleanRepresentation:
                        ffand = output[:self.logic[i].dims[1] // 4].all(1)
                        ffor = output[self.logic[i].dims[1] // 4: self.logic[i].dims[1] // 2].any(1)
                        ffand2 = torch.cat( [output[self.logic[i].dims[1] // 2:3*self.logic[i].dims[1] // 4, 0].unsqueeze(1),
                                 ~ output[self.logic[i].dims[1] // 2:3*self.logic[i].dims[1] // 4, 1].unsqueeze(1)], 1).all(1)
                        ffor2 = torch.cat( [output[3*self.logic[i].dims[1] // 4:, 0].unsqueeze(1),
                                        ~ output[3*self.logic[i].dims[1] // 4:, 1].unsqueeze(1)], 1).any(1)
                    else:
                        ffand = output[:self.logic[i].dims[1] // 4].prod(1)
                        ffor = 1.- (1. - output[self.logic[i].dims[1] // 4: self.logic[i].dims[1] // 2]).prod(1)
                        ffand2 = torch.cat( [output[self.logic[i].dims[1] // 2:3*self.logic[i].dims[1] // 4, 0].unsqueeze(1),
                                1. - output[self.logic[i].dims[1] // 2:3*self.logic[i].dims[1] // 4, 1].unsqueeze(1)], 1).prod(1)
                        ffor2 = 1. - (1. - torch.cat( [output[3*self.logic[i].dims[1] // 4:, 0].unsqueeze(1),
                                       1. - output[3*self.logic[i].dims[1] // 4:, 1].unsqueeze(1)], 1)).prod(1)

                    output = torch.cat([ffand, ffor, ffand2, ffor2])
                    output = output.permute(*range(1, len(f.shape)), 0)
                    if self.residual and self.input_dims[i] > 0:
                        if extract_rule:
                            print("residual")
                        output = torch.cat([inputs[i], output], dim=-1)
                    outputs.append(output)
                    # for each element p in path
                    # p=p // (4n) --> know the permutation, for example (3,2,1) meams permute(0,3,2,1,4)
                    # permute back-->e.g. a.permute(k[0],k[1],k[2]).permute(k[k0],k[k[1]],k[k[2]])==a
                    # p%4n, p \in [0,4n) now. expand, input[i], or reduce
                    # p \in [0,n)
                    # path is input[batch,training instance?,p]
        else:
            # print(inputs[2].shape) #[4,10,10,4][batch_size,#training instances,#training instances,number of prediction?IsSon,IsDaughter,IsFatherandIsMother?]
            for i in range(self.max_order + 1):
                # collect input f from group i-1, i and i+1.
                f = []
                findex = []
                if i > 0 and self.input_dims[i - 1] > 0:
                    n = inputs[i].size(1) if i == 1 else None
                    f.append(self.dim_expanders[i](inputs[i - 1], n))
                    #findex.extend(['e:p'+str(j) for j in range(inputs[i - 1].shape[-1])])
                if i < len(inputs) and self.input_dims[i] > 0:
                    f.append(inputs[i])
                    #findex.extend([':p'+str(j) for j in range(inputs[i].shape[-1])])
                if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                    f.append(self.dim_reducers[i](inputs[i + 1]))
                    #findex.extend(['rma:p' + str(j) for j in range(inputs[i + 1].shape[-1])])
                    #findex.extend(['rmi:p' + str(j) for j in range(inputs[i + 1].shape[-1])])
                if len(f) == 0:
                    output = None
                else:
                    f = torch.cat(f, dim=-1)
                    #findex = self.dim_perms[i].forward_index(f, findex)
                    f = self.dim_perms[i](f)
                    output, other_output = self.logic[i](f, findex)
                    update_dict_list(other_outputs, other_output)
                    # f is the input of self.logic[i], f=permutation([expand(inputs[i-1]),inputs[i],reduce(inputs[i+1])])
                    # expand: e.g. [4,10,4] to [4,10,10,4] by torch.expand
                    # reduce: if reduce.exists==true, e.g.[4,10,10_,4] to [4,10,8], stack(max(input0,dim=-2),min(input1,dim=-2)), input0,1 is input exclude [:,x,x,:]
                    #        else [4,10,10_,4] to [4,10,4], max(input,dim=-2)
                    # permutation: input[4,10,10,....,8]. dimension[0,1,2,3,....n+1]. permutatoins((1,...,n)), torch.permute(). [(1, 2), (2, 1)],[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
                    # permute r elements, the jth permutation is ? jth element in itertools.permutations((1,...,r)). calculate during initialization?

                    # put it at the beginning of forward after test

                    # in hasfather, we only have input[2]
                    # Generally, suppose we have inputs[i-1].shape=[batch_size,10,...,10,n],inputs[i].shape=[batch_size,10,...,10,n], (10^i)， inputs[i+1]
                    # f=permutation(shape(batch_size,10,...,10,n+n+2*n=4n))
                    # f.shape=(batch_size,10,...,10,4n*i!)

                if self.residual and self.input_dims[i] > 0:
                    output = torch.cat([inputs[i], output], dim=-1)
                outputs.append(output)

        return outputs, other_outputs


class DifferentiableLogicMachine(nn.Module):
    """Neural Logic Machine consists of multiple logic layers."""

    def __init__(
            self,
            depth,
            breadth,
            input_dims,
            output_dims,
            exclude_self=True,
            residual=False,
            io_residual=False,
            recursion=False,
            dlm_intern_params=None
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.residual = residual
        self.io_residual = io_residual
        self.recursion = recursion
        self.extract_path = False
        self.dlm_intern_params = dlm_intern_params
        self.allpath = []
        self.stacked_models = []
        self.is_stacked_model = False
        self.used_predicates_graph = []
        self.input_dims = input_dims
        self.exclude_self = exclude_self
        self.number_attribute = output_dims
        self.output_feature_axis = None
        self.pathFromPred = None
        self.track_duplicates = None
        self.track_duplicates_record = []
        self.used_inputs = []
        self.filter_inputs = []
        self.tail = False
        self.store_previous_tail = False

        assert not (self.residual and self.io_residual), \
            'Only one type of residual connection is allowed at the same time.'

        # element-wise addition for vector
        def add_(x, y):
            for i in range(len(y)):
                x[i] += y[i]
            return x

        self.layers = nn.ModuleList()
        current_dims = input_dims
        total_output_dims = [0 for _ in range(self.breadth + 1) ]  # for IO residual only
        for i in range(depth):
            # IO residual is unused.
            if i > 0 and io_residual:
                add_(current_dims, input_dims)
            # Not support output_dims as list or list[list] yet.
            layer = DLMLayer(breadth, current_dims, output_dims, i,
                               exclude_self, residual, 'l'+str(i), dlm_intern_params)
            current_dims = layer.output_dims
            if io_residual:
                add_(total_output_dims, current_dims)
            self.layers.append(layer)

        if io_residual:
            self.output_dims = total_output_dims
        else:
            self.output_dims = current_dims

    def update_tau(self, tau):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.update_tau(tau)

    def update_gumbel_noise(self, tau):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.update_gumbel_noise(tau)

    def update_dropout_prob(self, tau):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.update_dropout_prob(tau)

    def independant_noise_per_sample(self, b):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.independant_noise_per_sample = b

    def with_gumbel(self, b):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.with_gumbel = b

    def with_dropout(self, b):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.with_dropout = b

    def reset_weights(self):
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    l2.reset_weights()

    def extract_graph(self, output_feature_axis, pred, track_duplicates=False):
        self.output_feature_axis = output_feature_axis
        self.pathFromPred = pred.weight.argmax(-1)
        self.pathFromPred = self.pathFromPred[self.pathFromPred != self.number_attribute].unsqueeze(0)
        self.track_duplicates = track_duplicates
        self.eval()
        self.extract_path = True

    def continuous_relaxation(self):
        self.allpath = []
        self.used_predicates_graph = []
        self.used_inputs = []
        self.train()
        self.extract_path = False
        self.track_duplicates = False
        self.tail = False

    def fake_forward(self):
        nobj = 2
        nbatch = 1
        f = [None for i in range(self.breadth + 1)]
        for i, dim in enumerate(self.input_dims):
            if dim != 0:
                shape = [nbatch] + [nobj for _ in range(i)] + [dim]
                f[i] = torch.zeros(shape, dtype=torch.bool, device=next(self.parameters()).device, requires_grad=False)
        self(f)

    def display_distribution_extracted(self, pred, tau):
        for i, layer in enumerate(self.used_predicates_graph):
            print("LAYER ", i+1)
            for j, breadth in enumerate(layer):
                for index in breadth:
                    print((self.layers[i].logic[j].weight/tau).softmax(-1)[index].sort(descending=True)[0][:, :4].detach().cpu().numpy())
        print("LAST PRED")
        print((pred.weight.softmax(-1)/tau).sort(descending=True)[0][0, :, :4].detach().cpu().numpy())

    def extract_stacked(self, output_feature_axis, pred, new_param=dict(), remove_duplicates=True, onlyLastLayer=False,
                                removeOldUnused=False, with_tail=False, with_even=False, removeOldUnusedDeg=1,
                                store_previous_tail=False):

        if store_previous_tail:
            with_tail = True
        self.store_previous_tail = store_previous_tail
        if self.stacked_models:
            self.stacked_models[-1].store_previous_tail = False

        self.extract_graph(output_feature_axis, pred)
        if len(self.allpath) == 0 or len(self.used_predicates_graph) == 0:
            with torch.set_grad_enabled(False):
                self.fake_forward() #compute outputs shapes

        if with_tail:
            selected_preds = pred.weight.argmax(-1)[0]
            #twice same pred
            if (selected_preds[0] == selected_preds).all():
                with_tail = False
            #too much bias in the tail
            if (selected_preds == self.number_attribute).int().sum() >= (self.dlm_intern_params['atoms_per_rule'] - 1):
                with_tail = False
        if with_tail:
            self.tail = True

        if remove_duplicates:
            for (key, value) in self.track_duplicates_record.items():
                if (value == 0).any():
                    logger.info("duplicated detected at " + str(key) + "(depth, breadth, pred_index) with input " + str(value.argmin()))
                    self.used_predicates_graph[key[0]][key[1]].remove(key[2])

        if onlyLastLayer:
            for i in range(self.depth - 1):
                for j in range(self.breadth + 1):
                    self.used_predicates_graph[i][j] = set()

        if with_even:
            for i in range(self.depth - 1):
                if i % 2 == 1:
                    for j in range(self.breadth + 1):
                        self.used_predicates_graph[i][j] = set()

        self.filter_inputs = []
        if self.stacked_models and removeOldUnused:
            self.filter_inputs = [set(range(dims)) for dims in self.input_dims]
            #do not filter very first inputs
            first_model = self
            while len(first_model.stacked_models) != 0:
                first_model = first_model.stacked_models[0]

            for i in range(self.breadth + 1):
                self.filter_inputs[i] -= [set(range(dims)) for dims in first_model.input_dims][i]
                # nor used elements
                self.filter_inputs[i] -= self.used_inputs[i]
                first_model = self
                for _ in range(removeOldUnusedDeg - 1):
                    if len(first_model.stacked_models) == 0:
                        break
                    first_model = first_model.stacked_models[0]
                    self.filter_inputs[i] -= first_model.used_inputs[i]
                self.filter_inputs[i] = list(self.filter_inputs[i])

        self.is_stacked_model = True
        self.track_duplicates = False

        shapes = [sum([len(s[i]) for s in self.used_predicates_graph]) for i in range(self.breadth + 1)]
        input_dims = (np.array(self.input_dims) + np.array(shapes)).tolist()
        if with_tail:
            input_dims[output_feature_axis] += 1
        if self.filter_inputs:
            input_dims = (np.array(input_dims) - np.array([len(l) for l in self.filter_inputs])).tolist()
        D = dict(depth=self.depth, breadth=self.breadth, input_dims=input_dims,
                                              output_dims=self.number_attribute, exclude_self=self.exclude_self,
                                              residual=self.residual, io_residual=self.io_residual, recursion=self.recursion,
                                              dlm_intern_params=self.dlm_intern_params)
        D.update(new_param)
        new_model = DifferentiableLogicMachine(**D)
        new_model.to(next(self.parameters()).device)
        new_model.stacked_models = [self]
        #save some memories
        for l in self.layers:
            for l2 in l.logic:
                if l2 is not None:
                    del l2.weight

        return new_model, np.array(input_dims).sum()

    def forward(self, inputs, depth=None, extract_rule=False):
        previous_stacked_tail = None

        for i, prev_model in enumerate(self.stacked_models):
            if i == 0:
                f = inputs
            out_prev = prev_model(f, depth, extract_rule)
            f = out_prev[0]
            if 'previous_tail' in out_prev[1].keys():
                previous_stacked_tail = out_prev[1]['previous_tail']

        if self.stacked_models:
            for j in range(self.breadth + 1):
                if self.stacked_models[0].filter_inputs and len(self.stacked_models[0].filter_inputs[j]) > 0:
                    index = torch.ones(inputs[j].shape[-1], dtype=bool, device=inputs[j].device, requires_grad=False)
                    index[self.stacked_models[0].filter_inputs[j]] = False
                    inputs[j] = merge(inputs[j][..., index], f[j])
                else:
                    inputs[j] = merge(inputs[j], f[j])


        # depth: the actual depth used for inference
        if depth is None:
            depth = self.depth
        if not self.recursion:
            depth = min(depth, self.depth)
        if self.extract_path:
            allPaths = []
            self.eval()
        else:
            allPaths = [None for _ in range(depth)]
        if self.extract_path and len(self.allpath) == 0:
            # do a fake Forward to get the sizes we need to extract path backward
            inputsSize = []
            layers = []

            isResiduals = []
            isExpands = []
            isInputs = []
            isReduces = []
            sizesResidual = []
            sizesExpand = []
            sizesInput = []
            sizesReduce = []

            for i in range(len(inputs)):
                if inputs[i] is not None:
                    inputsSize.append(list(inputs[i].size()))
                else:
                    inputsSize.append(None)
            layer = None
            last_layer = None
            for i in range(depth+1):
                if i < depth:
                    if i > 0 and self.io_residual:
                        for j, inp in enumerate(inputs):
                            # f[j] = merge(f[j], inp)
                            if inputsSize[j] is None:
                                if inp is not None:
                                    inputsSize[j] = list(inp.size())
                            if inp is None:
                                continue
                            inputsSize[j][-1] += list(inp.size())[-1]

                    if self.recursion and i >= 3:
                        assert not self.residual
                        layer, last_layer = last_layer, layer
                    else:
                        last_layer = layer
                        layer = self.layers[i]
                        layers.append(layer)
                    if i==depth-1:
                        enable_feature_axis=True
                    else:
                        enable_feature_axis=False
                    inputsSize, isResidual, isExpanded, isInput, isReduced, sizeResidual, sizeExpand, sizeInput, sizeReduce, paths = layer.fakeForward(inputsSize, enable_feature_axis, self.output_feature_axis)

                if i == depth:
                    paths = self.pathFromPred
                isResiduals.append(isResidual)
                isExpands.append(isExpanded)
                isInputs.append(isInput)
                isReduces.append(isReduced)
                sizesResidual.append(sizeResidual)
                sizesExpand.append(sizeExpand)
                sizesInput.append(sizeInput)
                sizesReduce.append(sizeReduce)
                allPaths.append(paths)

            # now go backward
            for i in range(len(layers), -1, -1):
                residualusefulp = [set() for j in range(self.breadth + 1)]
                expandusefulp = [set() for j in range(self.breadth + 1)]
                inputusefulp = [set() for j in range(self.breadth + 1)]
                reduceusefulp = [set() for j in range(self.breadth + 1)]

                if i != len(layers):
                    for j in range(self.breadth + 1):
                        # p=p-negation*(self.logic[j].layer[0].weight.shape[-1]//2)
                        # p=p%(sizeExpand+sizeInput+sizeReduce)
                        # inputs[j - 1][...,p] or inputs[j][...,p-sizeExpand] or inputs[j+1][...,[(p-(sizeExpand+sizeInput))//2]]
                        # only need those inputs
                        # previous layer, output[j][...,j] is useless for example.
                        # torch.cat([ffand,ffor]).permute(...,0)
                        # for output[j][...,m], comes from output[j][m,...]
                        # means from ffand or ffor, f[m,[0,1],...] is useless
                        # means in the previous layer, weight[m,:] is useless
                        # back propagation: current layer set p --> modify previous
                        if (i == len(layers)-1 and j != self.output_feature_axis) or allPaths[i][j] is None:
                            continue
                        else:
                            pathset=set(allPaths[i][j].cpu().numpy().flatten())
                            for p in pathset:
                                if p == -1:
                                    continue
                                if p < sizesResidual[i][j]:
                                    assert isResiduals[i][j]
                                    residualusefulp[j].add(p)
                                    continue
                                else:
                                    p -= sizesResidual[i][j]
                                #if 2 * p >= layers[i].logic[j].weight.shape[-1] - sizesResidual[i][j]:
                                #    p -= (layers[i].logic[j].weight.shape[-1] - sizesResidual[i][j]) // 2
                                p = p % (sizesExpand[i][j] + sizesInput[i][j] + sizesReduce[i][j])
                                if p < sizesExpand[i][j]:
                                    assert isExpands[i][j]
                                    expandusefulp[j].add(p)
                                elif p < sizesInput[i][j] + sizesExpand[i][j] + sizesResidual[i][j]:
                                    assert isInputs[i][j]
                                    inputusefulp[j].add(p - sizesExpand[i][j])
                                else:
                                    assert isReduces[i][j]
                                    reduceusefulp[j].add((p - (sizesExpand[i][j] + sizesInput[i][j])) // 2)

                    if i > 0:
                        for j in range(self.breadth + 1):
                            if i==len(layers)-1 and j != self.output_feature_axis:
                                for m in range(allPaths[i][j].size(0)):
                                    allPaths[i][j][m, :] = -1

                            if j == 0:
                                if allPaths[i - 1][j] is not None:
                                    for m in range(allPaths[i - 1][j].size(0)):
                                        if (m not in inputusefulp[j]) and (m not in expandusefulp[j + 1]) and (m not in residualusefulp[j]):
                                            allPaths[i - 1][j][m, :] = -1
                            elif j > 0 and j < self.breadth:
                                if allPaths[i - 1][j] is not None:
                                    for m in range(allPaths[i - 1][j].size(0)):
                                        if (m not in expandusefulp[j + 1]) and (m not in inputusefulp[j]) and (m not in reduceusefulp[j - 1]) and (m not in residualusefulp[j]):
                                            allPaths[i - 1][j][m, :] = -1
                            else:
                                if allPaths[i - 1][j] is not None:
                                    for m in range(allPaths[i - 1][j].size(0)):
                                        if (m not in inputusefulp[j]) and (m not in reduceusefulp[j - 1]) and (m not in residualusefulp[j]):
                                            allPaths[i - 1][j][m, :] = -1
                else:
                    pathset = set(allPaths[i].cpu().numpy().flatten())
                    for j in range(self.breadth + 1):
                        for m in range(allPaths[i - 1][j].size(0)):
                            if m not in pathset:
                                allPaths[i - 1][j][m, :] = -1
                    j = self.output_feature_axis
                    for p in pathset:
                        if p != self.number_attribute: #remove bias
                            inputusefulp[self.output_feature_axis].add(p)

                usedPrevious = copy.deepcopy(inputusefulp)
                for ind in range(self.breadth + 1):
                    if ind + 1 < len(expandusefulp):
                        usedPrevious[ind] = usedPrevious[ind].union(expandusefulp[ind + 1])
                    if ind >= 0:
                        usedPrevious[ind] = usedPrevious[ind].union(reduceusefulp[ind - 1])
                if i != 0:
                    self.used_predicates_graph.append(usedPrevious)
                else:
                    self.used_inputs = usedPrevious

            self.allpath = allPaths
            self.used_predicates_graph = list(reversed(self.used_predicates_graph))
            self.track_duplicates_record = dict()
            for d, layer in enumerate(self.used_predicates_graph):
                for breadth, module in enumerate(layer):
                    for index_module in module:
                        if inputs[breadth] is not None:
                            self.track_duplicates_record[(d, breadth, index_module)] = torch.zeros(inputs[breadth].shape[-1], dtype=torch.int, requires_grad=False)
        elif self.extract_path and len(self.allpath) > 0:
            allPaths = self.allpath
        outputs = [None for _ in range(self.breadth + 1)]
        f = inputs

        layer = None
        last_layer = None

        other_outputs = {}
        for i in range(depth):
            if extract_rule:
                print("layer", i)
            if i > 0 and self.io_residual:
                for j, inp in enumerate(inputs):
                    f[j] = merge(f[j], inp)
            # To enable recursion, use scroll variables layer/last_layer
            # For weight sharing of period 2, i.e. 0,1,2,1,2,1,2,...
            if self.recursion and i >= 3:
                assert not self.residual
                layer, last_layer = last_layer, layer
            else:
                last_layer = layer
                layer = self.layers[i]

            f, other_output = layer(f, allPaths[i], extract_rule)
            update_dict_list(other_outputs, other_output)

            if self.io_residual:
                for j, out in enumerate(f):
                    outputs[j] = merge(outputs[j], out)
            if self.is_stacked_model:
                for j, out in enumerate(f):
                    if out is not None and len(self.used_predicates_graph[i][j]) > 0:
                        used_index = list(self.used_predicates_graph[i][j])
                        outputs[j] = merge(outputs[j], out[..., used_index])
                j = self.output_feature_axis
                if self.tail and f[j] is not None and i == depth - 1:
                    previous_tail = f[j][..., self.pathFromPred[0]].prod(-1).detach().unsqueeze(-1)
                    outputs[j] = merge(outputs[j], previous_tail)
                    if self.store_previous_tail:
                        previous_stacked_tail = previous_tail
                elif self.store_previous_tail:
                    previous_stacked_tail = f[j][..., self.pathFromPred[0]].prod(-1).detach().unsqueeze(-1)

            if self.track_duplicates:
                for j, out in enumerate(f):
                    if inputs[j] is not None:
                        if out is not None and len(self.used_predicates_graph[i][j]) > 0:
                            for module_index in list(self.used_predicates_graph[i][j]):
                                base = (inputs[j] == out[..., module_index].unsqueeze(-1))
                                while len(base.shape) > 2:
                                    base = base.all(1)
                                self.track_duplicates_record[(i, j, module_index)] += (base == False).int().sum(0).cpu()
        if not self.io_residual and not self.is_stacked_model:
            outputs = f

        if previous_stacked_tail is not None:
            other_outputs['previous_tail'] = previous_stacked_tail
        return outputs, other_outputs

    __hyperparams__ = (
        'depth',
        'breadth',
        'input_dims',
        'output_dims',
        'exclude_self',
        'io_residual',
        'residual',
        'recursion',
    )

    __hyperparam_defaults__ = {
        'exclude_self': True,
        'io_residual': False,
        'residual': False,
        'recursion': False,
        'atoms_per_rule': 2,
        'fuzzy_or': True,
        'add_negation': True,
    }

    __hyperparams_dlmintern__ = (
        'atoms_per_rule',
        'fuzzy_or',
        'add_negation',
    )

    @classmethod
    def make_nlm_parser(cls, parser, defaults, prefix=None):
        for k, v in cls.__hyperparam_defaults__.items():
            defaults.setdefault(k, v)

        if prefix is None:
            prefix = '--'
        else:
            prefix = '--' + str(prefix) + '-'

        parser.add_argument(
            prefix + 'depth',
            type=int,
            default=defaults['depth'],
            metavar='N',
            help='depth of the logic machine')
        parser.add_argument(
            prefix + 'breadth',
            type=int,
            default=defaults['breadth'],
            metavar='N',
            help='breadth of the logic machine')
        parser.add_argument(
            prefix + 'exclude-self',
            type='bool',
            default=defaults['exclude_self'],
            metavar='B',
            help='not allow multiple occurrence of same variable')
        parser.add_argument(
            prefix + 'io-residual',
            type='bool',
            default=defaults['io_residual'],
            metavar='B',
            help='use input/output-only residual connections')
        parser.add_argument(
            prefix + 'residual',
            type='bool',
            default=defaults['residual'],
            metavar='B',
            help='use residual connections')
        parser.add_argument(
            prefix + 'recursion',
            type='bool',
            default=defaults['recursion'],
            metavar='B',
            help='use recursion weight sharing')
        parser.add_argument(
            prefix + 'atoms-per-rule',
            type=int,
            default=defaults['atoms_per_rule'],
            metavar='N',
            help='number of atoms per rules for auxiliary predicates')
        parser.add_argument(
            prefix + 'fuzzy-or',
            type='bool',
            default=defaults['fuzzy_or'],
            metavar='B',
            help='allowing fuzzy OR')
        parser.add_argument(
            prefix + 'add-negation',
            type='bool',
            default=defaults['add_negation'],
            metavar='B',
            help='allowing negations')

        #kept the following for compatibility with nlm (might be remove in future)
        parser.add_argument(
            prefix + 'logic-hidden-dim',
            type=int,
            nargs='+',
            default=defaults['logic_hidden_dim'],
            metavar='N',
            help='hidden dim of the logic model')


    @classmethod
    def from_args(cls, input_dims, output_dims, args, prefix=None, **kwargs):
        if prefix is None:
            prefix = ''
        else:
            prefix = str(prefix) + '_'

        setattr(args, prefix + 'input_dims', input_dims)
        setattr(args, prefix + 'output_dims', output_dims)
        init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
        #init_params['extract_path'] = args.extract_path
        init_params.update(kwargs)

        dlm_intern_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams_dlmintern__}
        dlm_intern_params.update(kwargs)
        init_params['dlm_intern_params'] = dlm_intern_params

        return cls(**init_params)
