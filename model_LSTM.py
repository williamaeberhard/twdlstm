# twdlstm model_LSTM v0.7.1

if config['actout']=='ReLU':
    class Model_LSTM(torch.nn.Module):
        # def __init__(self, input_size, d_hidden, num_layers, output_size, z_size, z_fc_size):
        def __init__(self, input_size, d_hidden, num_layers, output_size):
            super().__init__()
            self.d_hidden = d_hidden
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=d_hidden,
                num_layers=num_layers,
                batch_first=True
            )
            # self.drop = torch.nn.Dropout(p=0.5)
            self.linear = torch.nn.Linear(
                # in_features=d_hidden + z_fc_size,
                in_features=d_hidden,
                out_features=output_size
            )
            # self.z_fc = torch.nn.Linear(z_size, z_fc_size)
            # self.z_act = torch.nn.Tanh()
            self.actout = torch.nn.ReLU()
        
        # def forward(self, x, z, hidden=None):
        def forward(self, x, hidden=None):
            if hidden is None:
                hidden = self.get_hidden(x)
            # x_lstm, hidden = self.lstm(x, hidden)
            # z_fc_out = self.z_act(self.z_fc(z)).unsqueeze(0).expand(x_lstm.shape[0], -1) # shape: (seq_len, z_fc_size)
            # x_concat = torch.cat([x_lstm.squeeze(0), z_fc_out], dim=1) # shape: (seq_len, d_hidden + z_fc_size)
            # x_out = self.actout(self.linear(x_concat))
            # return x_out, hidden
            x, hidden = self.lstm(x, hidden)
            x = self.actout(self.linear(x))
            return x, hidden
        
        def get_hidden(self, x):
            # second axis = batch size, i.e. x.shape[0] when batch_first=True
            hidden = (
                torch.zeros(
                    self.num_layers,
                    x.shape[0],
                    self.d_hidden,
                    device=x.device
                ),
                torch.zeros(
                    self.num_layers,
                    x.shape[0],
                    self.d_hidden,
                    device=x.device
                )
            )
            return hidden
elif config['actout']=='Softplus':
    class Model_LSTM(torch.nn.Module):
        # def __init__(self, input_size, d_hidden, num_layers, output_size, z_size, z_fc_size):
        def __init__(self, input_size, d_hidden, num_layers, output_size):
            super().__init__()
            self.d_hidden = d_hidden
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=d_hidden,
                num_layers=num_layers,
                batch_first=True
            )
            self.linear = torch.nn.Linear(
                # in_features=d_hidden + z_fc_size,
                in_features=d_hidden,
                out_features=output_size
            )
            # self.z_fc = torch.nn.Linear(z_size, z_fc_size)
            # self.z_act = torch.nn.Tanh()
            self.actout = torch.nn.Softplus()
        
        # def forward(self, x, z, hidden=None):
        def forward(self, x, hidden=None):
            if hidden is None:
                hidden = self.get_hidden(x)
            # x_lstm, hidden = self.lstm(x, hidden)
            # z_fc_out = self.z_act(self.z_fc(z)).unsqueeze(0).expand(x_lstm.shape[0], -1)  # shape: (seq_len, z_fc_size)
            # x_concat = torch.cat([x_lstm.squeeze(0), z_fc_out], dim=1) # shape: (seq_len, d_hidden + z_fc_size)
            # x_out = self.actout(self.linear(x_concat))
            # return x_out, hidden
            x, hidden = self.lstm(x, hidden)
            x = self.actout(self.linear(x))
            return x, hidden
        
        def get_hidden(self, x):
            # second axis = batch size, i.e. x.shape[0] when batch_first=True
            hidden = (
                torch.zeros(
                    self.num_layers,
                    x.shape[0],
                    self.d_hidden,
                    device=x.device
                ),
                torch.zeros(
                    self.num_layers,
                    x.shape[0],
                    self.d_hidden,
                    device=x.device
                )
            )
            return hidden
elif config['actout']=='Sigmoid':
    class Model_LSTM(torch.nn.Module):
        # def __init__(self, input_size, d_hidden, num_layers, output_size, z_size, z_fc_size):
        def __init__(self, input_size, d_hidden, num_layers, output_size):
            super().__init__()
            self.d_hidden = d_hidden
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=d_hidden,
                num_layers=num_layers,
                batch_first=True
            )
            self.linear = torch.nn.Linear(
                # in_features=d_hidden + z_fc_size,
                in_features=d_hidden,
                out_features=output_size
            )
            # self.z_fc = torch.nn.Linear(z_size, z_fc_size)
            # self.z_act = torch.nn.Tanh()
            self.actout = torch.nn.Sigmoid()
        
        # def forward(self, x, z, hidden=None):
        def forward(self, x, hidden=None):
            # if hidden is None:
            #     hidden = self.get_hidden(x)
            # x_lstm, hidden = self.lstm(x, hidden)
            # z_fc_out = self.z_act(self.z_fc(z)).unsqueeze(0).expand(x_lstm.shape[0], -1)  # shape: (seq_len, z_fc_size)
            # x_concat = torch.cat([x_lstm.squeeze(0), z_fc_out], dim=1) # shape: (seq_len, d_hidden + z_fc_size)
            # x_out = self.actout(self.linear(x_concat))
            # return x_out, hidden
            x, hidden = self.lstm(x, hidden)
            x = self.actout(self.linear(x))
            return x, hidden

# def get_hidden(self, x):
#     hidden = (
#         torch.zeros(
#             self.num_layers,
#             x.shape[0],
#             self.d_hidden,
#             device=x.device
#         ),
#         torch.zeros(
#             self.num_layers,
#             x.shape[0],
#             self.d_hidden,
#             device=x.device
#         )
#     )
#     return hidden

# model = Model_LSTM(i_size, h_size, nb_layers, o_size) # instantiate
# model.train() # print(model)
