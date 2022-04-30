import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from siamese_dataset import train_dataset , test_dataset , normalize
from siamese_model import FeatureExtractor , DistanceLayer
from siamese_loss import ContrastiveLoss


class siamese_trainer():

    def __init__(self,args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.normalize = args.normalize
        self.resume = args.resume
        self.use_log = args.log

        self.start_epoch = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}")


    def load_data(self):

        print("[!] Data Loading...")

        self.train_dataset = train_dataset(normalize() if self.normalize else None)
        
        self.train_loader = DataLoader(dataset = self.train_dataset,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       num_workers = 1)

        data_statistics = self.train_dataset.get_statistics()

        self.test_dataset = test_dataset(normalize() if self.normalize else None , data_statistics)

        self.test_loader = DataLoader(dataset = self.test_dataset,
                                      batch_size = self.batch_size,
                                      shuffle = False,
                                      num_workers = 1)

        # simple test
        # a , b = self.train_dataset[0]
        # print(type(a) , type(b))

        print("[!] Data Loading Done.")


    def setup(self):

        print("[!] Setup...")

        # define our model, loss function, and optimizer

        self.FeatureExtractor = FeatureExtractor().to(self.device)
        self.DistanceLayer = DistanceLayer().to(self.device)

        # actually DistanceLayer has no any parameters, 
        # so it is not necessary to update parameters,
        # however, we can still add its parameters to the list
        parameters = list(self.FeatureExtractor.parameters()) +\
                     list(self.DistanceLayer.parameters())
        
        if self.optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(parameters, lr=self.lr , weight_decay = 1e-4)
        else:
            self.optimizer = torch.optim.SGD(parameters, lr=self.lr , weight_decay = 1e-4)

        self.criterion = ContrastiveLoss().to(self.device)


        # load checkpoint file to resume training
        if self.resume:
            print(f"[!] Resume training from the file : {self.resume}")
            checkpoint = torch.load(self.resume)
            self.FeatureExtractor.load_state_dict(checkpoint['model_state'][0])
            self.DistanceLayer.load_state_dict(checkpoint["model_state"][1])
            try:
                self.start_epoch = checkpoint['epoch']
            except:
                pass

        if self.use_log:
            self.log_writer = SummaryWriter('logs')



        print("[!] Setup Done.")


    def train(self):

        print("[!] Model training...")

        n_total_steps = len(self.train_loader)
        n_total_samples = self.batch_size * n_total_steps

        for epoch in range(self.epochs):
            total_loss = 0
            running_loss = 0
            total_accuracy = 0
        
            for i , (data_pairs , targets) in enumerate(self.train_loader):

                # access data
                images1 = data_pairs[:,0].to(self.device)
                images2 = data_pairs[:,1].to(self.device)

                targets = targets.to(self.device)

                # model feedforward
                features1 = self.FeatureExtractor(images1)
                features2 = self.FeatureExtractor(images2)
                distance = self.DistanceLayer(features1 , features2)

                # compute loss
                loss = self.criterion(distance , targets)

                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # record loss and accuracy
                total_loss += loss.item() / n_total_steps
                running_loss += loss.item() / n_total_steps
                total_accuracy += ((distance <= 0.5) == targets).sum().item() / n_total_samples

                

                if (i+1) % 100 == 0:
                    print(f"[!] Epoch : [{epoch+1}], step : [{i+1} / {n_total_steps}], Running Loss: {running_loss:.6f}")
                    running_loss = 0


            # per-epoch logging
            print("------------------------------------------")
            print(f"[!] Epoch : [{epoch+1}/{self.epochs}] , Loss: {total_loss:.6f}, Accuracy: {total_accuracy:.4f}\n")
            if self.use_log:
                self.log_writer.add_scalar('training loss', total_loss, epoch)
                self.log_writer.add_scalar('training acc', total_accuracy, epoch)

        if self.use_log:
            self.log_writer.close()

        print("[!] Training Done.\n")

    
    def save(self):

        print("[!] Model saving...")

        checkpoint = {
                       "model_state": [self.FeatureExtractor.state_dict() , self.DistanceLayer.state_dict()],
                     }

        torch.save(checkpoint , "checkpoint.pth")
    
        print("[!] Saving Done.")



    def test(self):

        print("[!] Model testing...")

        total_loss , test_accuracy = 0 , 0
        n_total_steps = len(self.test_loader)
        n_total_samples = self.batch_size * n_total_steps

        for i , (data_pairs , targets) in enumerate(self.test_loader):

            # access data
            images1 = data_pairs[:,0].to(self.device)
            images2 = data_pairs[:,1].to(self.device)

            targets = targets.to(self.device)

            # model feedforward
            features1 = self.FeatureExtractor(images1)
            features2 = self.FeatureExtractor(images2)
            distance = self.DistanceLayer(features1 , features2)

            # compute loss
            loss = self.criterion(distance , targets)

            # record loss and accuracy
            total_loss += loss.item() / n_total_steps
            test_accuracy += ((distance <= 0.5) == targets).sum().item() / n_total_samples

        print(f"[!] Testing, Loss: {total_loss:.6f}, Accuracy: {test_accuracy:.4f}\n")