import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers
from torchdrug.layers import MeanReadout, SumReadout


class GeneralizedRelationalConv(layers.MessagePassingBase):
    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        query_input_dim,
        message_func="distmult",
        aggregate_func="pna",
        layer_norm=False,
        activation="relu",
        dependent=True,
        set_boundary=False,
        rgcn=False,
        num_bases=None,
        has_readout=False,
        readout_type="mean",
        query_specific_readout=False,
    ):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.set_boundary = set_boundary
        self.rgcn = rgcn
        self.num_bases = num_bases

        self.has_readout = has_readout
        self.readout_type = readout_type
        self.query_specific_readout = query_specific_readout
        if has_readout:
            if query_specific_readout:
                self.outward_readout = nn.Linear(input_dim, output_dim)
                self.inward_readout = nn.Linear(input_dim, output_dim)
                self.transform_readout = nn.Linear(input_dim, output_dim)
            else:
                if self.readout_type == "mean":
                    self.readout = MeanReadout()
                elif self.readout_type == "sum":
                    self.readout = SumReadout()
                else:
                    raise NotImplementedError
                self.transform_readout = nn.Linear(input_dim, output_dim)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if rgcn:
            # Message types 3
            if num_bases is None:
                self.weight = Parameter(
                    torch.Tensor(num_relation, input_dim, input_dim)
                )
                torch.nn.init.xavier_uniform_(self.weight)
            else:
                self.weight = Parameter(torch.Tensor(num_bases, input_dim, input_dim))
                self.comp = Parameter(torch.Tensor(num_relation, num_bases))
                torch.nn.init.xavier_uniform_(self.weight)
                torch.nn.init.xavier_uniform_(self.comp)
        else:
            if dependent:
                # Message types 1
                self.relation_linear = nn.Linear(
                    query_input_dim, num_relation * input_dim
                )
            else:
                # Message types 2
                self.relation = nn.Embedding(num_relation, input_dim)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation
        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.set_boundary:
            boundary = graph.boundary
        else:
            boundary = input
        node_input = input[node_in]
        if self.rgcn:
            if self.num_bases is None:
                weight = self.weight
            else:
                weight = (self.comp @ self.weight.view(self.num_bases, -1)).view(
                    self.num_relation, self.input_dim, self.input_dim
                )
            message = torch.bmm(node_input, weight[relation])
        else:
            if self.dependent:
                relation_input = self.relation_linear(graph.query).view(
                    batch_size, self.num_relation, self.input_dim
                )
            else:
                relation_input = self.relation.weight.expand(batch_size, -1, -1)
            relation_input = relation_input.transpose(0, 1)
            edge_input = relation_input[relation]
            if self.message_func == "transe":
                message = edge_input + node_input
            elif self.message_func == "distmult":
                message = edge_input * node_input
            elif self.message_func == "rotate":
                node_re, node_im = node_input.chunk(2, dim=-1)
                edge_re, edge_im = edge_input.chunk(2, dim=-1)
                message_re = node_re * edge_re - node_im * edge_im
                message_im = node_re * edge_im + node_im * edge_re
                message = torch.cat([message_re, message_im], dim=-1)
            else:
                raise ValueError("Unknown message function `%s`" % self.message_func)
        message = torch.cat([message, boundary])

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat(
            [node_out, torch.arange(graph.num_node, device=graph.device)]
        )
        edge_weight = torch.cat(
            [graph.edge_weight, torch.ones(graph.num_node, device=graph.device)]
        )
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(
                message * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )
        elif self.aggregate_func == "mean":
            update = scatter_mean(
                message * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )
        elif self.aggregate_func == "max":
            update = scatter_max(
                message * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(
                message * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )
            sq_mean = scatter_mean(
                message**2 * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )
            max = scatter_max(
                message * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )[0]
            min = scatter_min(
                message * edge_weight, node_out, dim=0, dim_size=graph.num_node
            )[0]
            std = (sq_mean - mean**2).clamp(min=self.eps).sqrt()
            features = torch.cat(
                [
                    mean.unsqueeze(-1),
                    max.unsqueeze(-1),
                    min.unsqueeze(-1),
                    std.unsqueeze(-1),
                ],
                dim=-1,
            )
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1
            )
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        return super(GeneralizedRelationalConv, self).message_and_aggregate(
            graph, input
        )

    def combine(self, input, update, graph, r_index=None):
        if self.has_readout:
            if self.query_specific_readout:
                ## Here we will aggregate all the rep with the specific query
                edge_list = graph.edge_list
                tail_read_out, head_read_out = self.calculate_read_out(
                    edge_list, input, r_index, self.readout_type
                )
                output = self.linear(torch.cat([input, update], dim=-1))
                tail_readout_update = self.outward_readout(tail_read_out)
                output += tail_readout_update
                head_readout_update = self.inward_readout(head_read_out)
                output += head_readout_update

            else:
                readout_update = self.readout(graph, input)
                readout_update = readout_update.repeat((input.size(0), 1, 1))
                output = self.linear(torch.cat([input, update], dim=-1))
                output += self.transform_readout(readout_update)
        else:
            output = self.linear(torch.cat([input, update], dim=-1))

        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, input, r_index=None):
        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(
                self._message_and_aggregate, *graph.to_tensors(), input
            )
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update, graph, r_index)
        return output

    def calculate_read_out(self, edge_list, input, r_index, readout_type):
        num_node, batch_size, dimension = input.shape

        # Create a mask to identify the relevant edge types in r_index
        r_index_expanded = r_index.unsqueeze(0).expand(edge_list.size(0), -1)
        edge_type_mask = edge_list[:, 2].unsqueeze(1) == r_index_expanded

        # Now create two node masks, one for tail and one for head; TODO: rename and swap the order
        tail_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        head_read_out_val = torch.zeros(1, batch_size, dimension, device=self.device)

        # TODO: Need to be further optimized by using torch masked sum
        for i in range(batch_size):
            # Find tail and head indices for each edge type
            tail_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)
            head_node_mask = torch.zeros(num_node, dtype=torch.bool, device=self.device)

            # tail_indices: all the source nodes with relation type q
            tail_indices = edge_list[edge_type_mask[:, i], 0]

            # head_indices: all the target node with relation type q
            head_indices = edge_list[edge_type_mask[:, i], 1]

            # Update the tail and head node masks
            tail_node_mask[tail_indices] = True
            head_node_mask[head_indices] = True

            # we read out node with only the value with input
            tail_masked_input = input[:, i, :] * tail_node_mask.unsqueeze(1).float()
            head_masked_input = input[:, i, :] * head_node_mask.unsqueeze(1).float()

            if readout_type == "sum":
                head_read_out_batch = head_masked_input.sum(dim=0)
                tail_read_out_batch = tail_masked_input.sum(dim=0)
            elif readout_type == "mean":
                head_read_out_batch = head_masked_input.mean(dim=0)
                tail_read_out_batch = tail_masked_input.mean(dim=0)
            else:
                raise NotImplementedError

            tail_read_out_val[:, i, :] = tail_read_out_batch.unsqueeze(0).unsqueeze(0)
            head_read_out_val[:, i, :] = head_read_out_batch.unsqueeze(0).unsqueeze(0)

        tail_read_out = tail_read_out_val.repeat(num_node, 1, 1)
        head_read_out = head_read_out_val.repeat(num_node, 1, 1)

        return tail_read_out, head_read_out
