%% ==============================
%        TESTING ONLY
% ===============================

clc; clear;

% 1️⃣ Load trained model
load('MnistConv.mat');   % loads W1, W5, Wo

% 2️⃣ Load MNIST test dataset
ImagesTest = LoadMNISTimages('DATASET/t10k-images.idx3-ubyte');
LabelsTest = LoadMNISTlabels('DATASET/t10k-labels.idx1-ubyte');

ImagesTest = reshape(ImagesTest, 28, 28, []);
LabelsTest(LabelsTest == 0) = 10;

% 3️⃣ Select full test set (recommended)
X = ImagesTest;      
D = LabelsTest;

N = length(D);
acc = 0;

%4️⃣ Forward pass only (NO backprop)

for k = 1:N
    
    x = X(:,:,k);

    % ----- Forward propagation -----
    y1 = Conv(x, W1);          % 24x24x32
    y2 = ReLU(y1);
    y3 = Pool(y2);             % 12x12x32
    y4 = reshape(y3, [], 1);   % 4608x1
    
    v5 = W5 * y4;              % 100x1
    y5 = ReLU(v5);
    
    v = Wo * y5;               % 10x1
    y = Softmax(v);
    % --------------------------------

    [~, pred] = max(y);

    if pred == D(k)
        acc = acc + 1;
    end
end

accuracy = acc / N;

fprintf('Test Accuracy = %.4f\n', accuracy);