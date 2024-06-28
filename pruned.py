import torch
import torch.nn as nn
import timm
import torch_pruning as tp
import torch.optim as optim


model = timm.create_model(model_name='resnet50', pretrained=True, in_chans=3, features_only=True)

torch.save(model, 'resnet50.pth')

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

imp = tp.importance.GroupTaylorImportance()

ignored_layers = [model.conv1,model.layer1[2].conv3,model.layer2[3].conv3,model.layer3[5].conv3,model.layer4[2].conv3]

example_inputs = torch.randn(1, 3, 224, 224)

pruner = tp.pruner.MetaPruner(
    model=model,
    example_inputs=example_inputs,
    importance=imp,
    pruning_ratio=0.5,
    ignored_layers=ignored_layers
)

# 3. Prune & finetune the model
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
print('base_macs = ' + str(base_macs))
print('base_nparams = ' + str(base_nparams))
if isinstance(imp, tp.importance.GroupTaylorImportance):
    # Taylor expansion requires gradients for importance estimation
    outputs = model(example_inputs)
    loss = 0
    target = [torch.randn(1, 64, 112, 112), torch.randn(1, 256, 56, 56), torch.randn(1, 512, 28, 28),
              torch.randn(1, 1024, 14, 14), torch.randn(1, 2048, 7, 7)]
    for output, tgt in zip(outputs, target):
        loss += criterion(output, tgt)
    loss.backward()
pruner.step()
macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
print('macs = ' + str(macs))
print('nparams = ' + str(nparams))
print('MACS_PRUNED_RATE='+str(100*(1-macs/base_macs))+'%')
print('PARAMS_PRUNED_RATE='+str(100*(1-nparams/base_nparams))+'%')
new_output = model(example_inputs)

torch.save(model, 'resnet50_pruned.pth')