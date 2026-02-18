%order{ loadimage, loadlabels, relu, softmax, Conv, pool, MnistConv, rng, TestMnist, display, plot }

% Fina script = to be run
% initializes + calls TRAINING(to train on data) + TESTs accuracy of CNN









% 1. TRAINING PART    ->Load TRAINING data (60k)
ImagesTrain = LoadMNISTimages('DATASET/train-images.idx3-ubyte');
ImagesTrain = reshape(ImagesTrain, 28, 28, []);

LabelsTrain = LoadMNISTlabels('DATASET/train-labels.idx1-ubyte');
LabelsTrain(LabelsTrain == 0) = 10;

rng(1);

% 2. Initialize Weights
W1 = 1e-2*randn([5 5 32]) ;                                  % size small-nos more {beore[9 9 20]}
W5 = (2*rand (100, 4608) - 1) * sqrt(6) / sqrt(100 + 4608);  %change size with W1
Wo = (2*rand( 10, 100) - 1) * sqrt (6) / sqrt ( 10 + 100);

X = ImagesTrain;
D = LabelsTrain;

%3. train
for epoch = 1:5  %Accuracy is 0.972500
     epoch           
    [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D) ;
end
                %   8->0.954   11->0.96 13->0.96  15->0.95
                %  More epochs → better learning (up to a limit)
                %  one complete pass of entire training dataset through CNN



% 4. Save Trained Model
save('MnistConv.mat') ;




% TESTING PART
ImagesTest = LoadMNISTimages('DATASET/t10k-images.idx3-ubyte');
LabelsTest = LoadMNISTlabels('DATASET/t10k-labels.idx1-ubyte');

ImagesTest = reshape(ImagesTest, 28, 28, []);
LabelsTest(LabelsTest == 0) = 10;

X = ImagesTest(:, :, 8001:10000);  % taking (images 8001–10000)of test set
D = LabelsTest(8001:10000);

acc = 0;
N = length (D) ;

    for k = 1:N
    
        x = X(:, :, k);
        
        y1 = Conv(x, W1) ;
        y2 = ReLU(y1);
        y3 = Pool(y2);
        
        y4 = reshape(y3, [], 1);
        v5 = W5*y4;
        y5 = ReLU(v5);
        
        v = Wo*y5;
        y = Softmax(v);
        
        [~, i] = max (y);
        
        if i == D(k)
          acc = acc + 1;
        
         end
    end


acc = acc/N;

fprintf('Accuracy is %f\n',acc)


%The trained neural network is the set of learned weights:

%   [ W1, W5, Wo] in MnistConv.mat file //Accuracy is 0.972500 now

%    W1 → Convolution Layer filters
%    W5 → Hidden Fully Connected Layer weights
%    Wo → Output Layer Weights

%   MNIST = 28×28 handwritten digit dataset


% INCREASE ACCURACY- 
% tune learning rate (apha)
%  tune epoch (no of times training ) - to  limit
% more no of conv filters/kernels, small sizes
% more no of CONV layers 
%