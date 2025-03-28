import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Adicionado Dropout após a segunda ReLU
        )

    def forward(self, x):
        return self.conv(x)
    
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET
        for feature in features:
            dropout_rate = 0.1 if feature <= 128 else 0.2  # Taxa maior para camadas mais profundas
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature
            
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            dropout_rate = 0.1 if feature <= 128 else 0.2  # Taxa maior para camadas mais profundas
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate))
            
        # Bottleneck com Dropout mais alto
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=0.3)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Chama a inicialização dos pesos
        self.initialize_weights()
        

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:  # Só inicializa se houver viés
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

def test():
    # x = torch.randn((3, 1, 161, 161))  # Harder test
    x = torch.randn((1, 3, 256, 256))
    model = UNET(in_channels=3, out_channels=1) # atenção in channels está 1
    preds = model(x)
    print(preds.shape)  # Dimensões -> [3, 1, 161, 161]
    print(x.shape)      # Dimensões -> [3, 1, 161, 161]
    # assert preds.shape == x.shape
    # for name, param in model.named_parameters():
        # print(name, param.data.mean(), param.data.std(unbiased=False))

# Teste básico
if __name__ == "__main__":
    test()