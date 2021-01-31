import torch.nn as nn
import torch.nn.functional as F

## Define the NN architecture
class MLPClassifier(nn.Module):
    def __init__(self, args):
        
        super(MLPClassifier, self).__init__()
        self.classifier= nn.Sequential(
            # linear layer (n_hidden -> hidden_2)
            nn.Linear(30, 256),
            nn.ReLU(),
            # linear layer (n_hidden -> 256)
            nn.Linear(256, 256),
            nn.ReLU(),            
            # dropout layer (p=0.2)
            # dropout prevents overfitting of data
            nn.Dropout(0.2),
            nn.Linear(256, args.nc),
            # nn.Sigmoid()
        )
        
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 30)
        # add hidden layer, with relu activation function
        x = self.classifier(x)
        return x



def numel(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def main():
    import torch
    m = MLPClassifier()
    print(m)
    x = torch.randn(1, 30, 1)
    with torch.no_grad():
        y = m(x)
        print(y.size())
        print(y)

    print(numel(m))
if __name__ == '__main__':
    main()