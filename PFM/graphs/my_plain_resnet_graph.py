from graphs.base_graph import BIGGraph, NodeType


class MyPlainResNetGraph(BIGGraph):

    def __init__(self, model,
                 shortcut_name='shortcut',
                 layer_name='segments.',
                 head_name='fc',
                 num_layers=3):
        super().__init__(model)

        self.shortcut_name = shortcut_name
        self.layer_name = layer_name
        self.num_layers = num_layers
        self.head_name = head_name

    def add_basic_block_nodes(self, name_prefix, input_node):
        shortcut_prefix = name_prefix + f'.{self.shortcut_name}'
        shortcut_output_node = input_node
        if shortcut_prefix in self.named_modules and len(self.get_module(shortcut_prefix)) > 0:
            # There's a break in the skip connection here, so add a new prefix
            input_node = self.add_nodes_from_sequence('', [NodeType.PREFIX], input_node)

            shortcut_output_node = self.add_nodes_from_sequence(
                name_prefix=shortcut_prefix,
                list_of_names=['0'],
                input_node=input_node
            )

        skip_node = self.add_nodes_from_sequence(
            name_prefix=name_prefix,
            list_of_names=[
                'conv1', NodeType.PREFIX, 'conv2', NodeType.SUM],
            input_node=input_node
        )

        self.add_directed_edge(shortcut_output_node, skip_node)

        return skip_node

    def add_layer_nodes(self, name_prefix, input_node):
        source_node = input_node
        for layer_index, block in enumerate(self.get_module(name_prefix)):
            source_node = self.add_basic_block_nodes(name_prefix+f'.{layer_index}', source_node)
        return source_node

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        input_node = self.add_nodes_from_sequence('', ['conv1'], input_node, sep='')

        for i in range(0, self.num_layers):
            input_node = self.add_layer_nodes(f'{self.layer_name}{i}', input_node)

        input_node = self.add_nodes_from_sequence('',
            [NodeType.PREFIX, 'avg_pool', self.head_name, NodeType.OUTPUT], input_node, sep='')

        return self


def my_plain_resnet20(model):
    return MyPlainResNetGraph(model)


if __name__ == '__main__':
    # unit test, nice
    # call from root directory with `python -m "graphs.resnet_graph"`
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torchvision.models.resnet as resnet
    import models.resnets as resnet2

    from model_merger import ModelMerge
    from matching_functions import match_tensors_identity, match_tensors_zipit
    from copy import deepcopy

    data_x = torch.rand(4, 3, 224, 224)
    data_y = torch.zeros(4)

    dataset = TensorDataset(data_x, data_y)
    dataloader = DataLoader(dataset, batch_size=4)

    model = resnet2.resnet20().eval()
    state_dict = model.state_dict()

    print(model)

    model3 = resnet2.resnet20().eval()

    graph1 = resnet20(deepcopy(model)).graphify()
    graph2 = resnet20(deepcopy(model)).graphify()

    merge = ModelMerge(graph1, graph2)
    merge.transform(model3, dataloader, transform_fn=match_tensors_zipit)

    graph1.draw(nodes=range(20))
    graph1.draw(nodes=range(len(graph1.G)-20, len(graph1.G)))

    print(model.eval().cuda()(data_x.cuda()))

    print(merge(data_x.cuda()))
