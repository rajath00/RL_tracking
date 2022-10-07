from torch import optim

class Train:

    def __init__(self,model, optimizer, loss_fn, num_epochs, train_batch):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        pass

    def training_loop(self):
        pass

    def train_epoch(self,input):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.forward(input).reshape(-1)
        loss = self.loss_fn(z, labels.float())
        loss.backward()
        self.optimizer.step()
