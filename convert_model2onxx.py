import torch
import torch.nn as nn
import torch.onnx as onnx
import torch.nn.functional as F
import argparse
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Parte descendente da UNET
        for feature in features:
            dropout_rate = 0.1 if feature <= 128 else 0.2
            self.downs.append(DoubleConv(in_channels, feature, dropout_rate))
            in_channels = feature

        # Parte ascendente da UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            dropout_rate = 0.1 if feature <= 128 else 0.2
            self.ups.append(DoubleConv(feature*2, feature, dropout_rate))

        # Gargalo com taxa de dropout mais alta
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_rate=0.3)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Inicializa os pesos
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

            # if x.size()[2:] != skip_connection.size()[2:]: # retirado por causa do warning no onxx
            x = F.interpolate(x, size=skip_connection.size()[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def main():

	# para ler oa argumentos do terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_path", type=str, required=True,
                        help="Caminho para o arquivo state_dict (.pth.tar)")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Caminho para salvar o arquivo ONNX")
    args = parser.parse_args()

    if not os.path.exists(args.state_dict_path):
        raise FileNotFoundError(f"Arquivo state_dict não encontrado em {args.state_dict_path}")

    model = UNET()
    checkpoint = torch.load(args.state_dict_path, map_location=torch.device('cpu'))  # Carrega o arquivo .pth.tar
    print(checkpoint.keys() if isinstance(checkpoint, dict) else "Apenas state_dict")
    # Tenta carregar o state_dict, lidando com diferentes formatos
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Caso o arquivo contenha apenas o state_dict
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 256)
    # onnx.export(model, dummy_input, args.onnx_path,
    #             input_names=["input"], output_names=["output"], verbose=True)

    torch.onnx.export(model, dummy_input, args.onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
                      verbose=True)

    print(f"Modelo exportado para {args.onnx_path}")

if __name__ == "__main__":
    main()



# python convert_model2onnx.py --state_dict_path "model.pth.tar" --onnx_path "modelo.onnx"