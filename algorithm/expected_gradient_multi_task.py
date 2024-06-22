#!/usr/bin/env python
import functools
import operator
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers.
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]

            params   indices   output

            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)


class AttributionPriorExplainer(object):
    def __init__(self, background_dataset, batch_size, random_alpha=True, k=1, scale_by_inputs=True):
        self.random_alpha = random_alpha
        self.k = k
        self.scale_by_inputs = scale_by_inputs
        self.batch_size = batch_size
        self.ref_set = background_dataset
        self.ref_sampler = DataLoader(
            dataset=background_dataset,
            batch_size=batch_size * k,
            shuffle=True,
            drop_last=True)
        return

    def _get_ref_batch(self, k=None):
        return next(iter(self.ref_sampler))[0].float()

    def _get_samples_input(self, input_tensor, reference_tensor):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions.
            reference_tensor: A tensor of shape (batch, k, ...) where ...
                indicates dimensions, and k represents the number of background
                reference samples to draw per input in the batch.
        Returns:
            samples_input: A tensor of shape (batch, k, ...) with the
                interpolated points between input and ref.
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)

        batch_size = reference_tensor.size()[0]
        k_ = reference_tensor.size()[1]

        # Grab a [batch_size, k]-sized interpolation sample
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_).uniform_(0, 1).to(DEFAULT_DEVICE)
        else:
            if k_ == 1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for i in range(batch_size)]).to(DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0, 1, k_) for i in range(batch_size)]).to(DEFAULT_DEVICE)

        shape = [batch_size, k_] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor

        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult

        # A fine Affine Combine
        samples_input = end_point_input + end_point_ref
        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd

    def _get_grads(self, samples_input, model, sparse_labels=None):
        # samples_input.requires_grad = True
        samples_input = samples_input.detach().clone().requires_grad_(True)

        # grad_tensor = torch.zeros(samples_input.shape).float().to(DEFAULT_DEVICE)

        # for i in range(self.k):
        # particular_slice = samples_input[:,i]
        # batch_output = model(particular_slice)
        # # should check that users pass in sparse labels
        # # Only look at the user-specified label
        # if batch_output.size(1) > 1:
        # sample_indices = torch.arange(0,batch_output.size(0)).to(DEFAULT_DEVICE)
        # indices_tensor = torch.cat([
        # sample_indices.unsqueeze(1),
        # sparse_labels.unsqueeze(1)], dim=1)
        # batch_output = gather_nd(batch_output, indices_tensor)

        # model_grads = grad(
        # outputs=batch_output,
        # inputs=particular_slice,
        # grad_outputs=torch.ones_like(batch_output).to(DEFAULT_DEVICE),
        # create_graph=True)
        # grad_tensor[:,i,:] = model_grads[0]

        # Accomodate for multitask learning use cases

        # Get the number of tasks
        temp_output = model(samples_input[:, 0])
        grad_tensors = [torch.zeros(samples_input.shape).float().to(DEFAULT_DEVICE) for _ in range(len(temp_output))]

        # For each task, do below
        for i in range(self.k):
            # Should store the input instead of getting the slice again
            # Otherwise, the grad() cannot recognize the input
            particular_slice = samples_input[:, i]
            batch_outputs = model(particular_slice)
            tasks_grads = []
            for idx in range(len(batch_outputs)):
                batch_output = batch_outputs[idx]
                if batch_output.size(1) > 1:
                    sample_indices = torch.arange(0, batch_output.size(0)).to(DEFAULT_DEVICE)
                    indices_tensor = torch.cat([
                        sample_indices.unsqueeze(1),
                        sparse_labels.unsqueeze(1)], dim=1)
                    batch_output = gather_nd(batch_output, indices_tensor)

                model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(DEFAULT_DEVICE),
                    create_graph=True)
                grad_tensors[idx][:, i, :] = model_grads[0]

        return grad_tensors

    def shap_values(self, model, input_tensor, sparse_labels=None):
        """
        Calculate expected gradients approximation of Shapley values for the
        sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
        """
        reference_tensor = self._get_ref_batch()
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
            self.batch_size,
            self.k,
            *(shape[1:])).to(DEFAULT_DEVICE)
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)

        # Accomodate for multitask learning

        # mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        # expected_grads = mult_grads.mean(1)

        mult_grads_list = [samples_delta * grad_tensor[i] if self.scale_by_inputs else grad_tensor[i] for i in
                           range(len(grad_tensor))]
        expected_grads_list = [mult_grads_list[i].mean(1) for i in range(len(mult_grads_list))]

        # Take average of all tasks
        expected_grads = sum(expected_grads_list) / len(expected_grads_list)

        #Armin Change: return list of expected grad for each ask instead of the one Jingyang returned
        return expected_grads_list # expected_grads

