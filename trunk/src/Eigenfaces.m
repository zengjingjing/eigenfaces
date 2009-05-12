%%
%% Master MVA
% Object Recognition and Artificial Vision
% Assignment 3
% PCA-based face recognition
%
% @author: Jean-Baptiste FIOT
%%

clear all; 
clc;

%% Parameters
display_results = 1;
various_K = 1:300;
% various_K=100;
if size(various_K,2)>1
    display_results = 0;
end
classifier_to_use = 1;
% -> Set classifier_to_use to 1 to use Euclidian distance on low-dimension
% projection space. (Fastest method!!)
% -> Set classifier_to_use to 2 to use NNC on reconstruted pictures

%%

fprintf('Loading data...\n');
load('ORL_32x32.mat'); % matrix with face images (fea) and labels (gnd)
load('train_test_orl.mat'); % training and test indices (trainIdx, testIdx)
fea = double(fea / 255);

if display_results
    display_faces(fea,10,10);
    title('Face data');
end

% partition the data into training and test subset
n_train = size(trainIdx,1);
n_test = size(testIdx,1);
train_data = fea(trainIdx,:);
train_label = gnd(trainIdx,:);
test_data = fea(testIdx,:);
test_label = gnd(testIdx,:);

fprintf('Running PCA...\n');
mean_face = mean(train_data);
if display_results
    figure;
    imshow(reshape(mean_face, [32,32]));
    title('Mean face');
end

train_data_centered = train_data - repmat(mean_face, [n_train,1]);
if display_results
    figure;
    display_faces(train_data_centered,10,10); 
    title ('Data centered');
end

[components, score, latent] = princomp(train_data_centered);% find principal components 
if display_results
    figure;
    display_faces(components,10,10); 
    title ('Top principal components');
end

classification_rate = [];

tic;
for K=various_K
   
    train_data_pca = train_data_centered * components(1:K,:)'; % low-dim coefficients for training data (projection onto components)
    train_data_reconstructed = train_data_pca * components(1:K,:);% high-dimensional faces reconstructed from the low-dim coefficients

    if display_results
        figure;
        display_faces(train_data_reconstructed,10,10); 
        title (sprintf('Reconstructed training data with K=%d',K));
    end


    test_data_centered = test_data - repmat(mean_face, [n_test,1]);

%     fprintf('Projecting test data...\n');
    test_data_pca = test_data_centered * components(1:K,:)';% low-dim coefficients for test data
    test_data_reconstructed = test_data_pca * components(1:K,:); % high-dimensional reconstructed test faces
%     fprintf('Running nearest-neighbor classifier...\n');


    if classifier_to_use == 1
        [nn_ind, estimated_label] = EuclDistClassifier(train_data_pca,train_label,test_data_pca);
    elseif classifier_to_use == 2
        [nn_ind, estimated_label] = NNclassifier(train_data_reconstructed,train_label,test_data_reconstructed);% output of nearest-neighbor classifier:
        % nearest neighbor training indices for each training point and 
        % estimated labels (corresponding to labels of the nearest
        % neighbors)
    else
        display('Set classifier_to_use parameter to 1 or 2)');
    end

    rate = sum(estimated_label == test_label)/n_test;
    classification_rate = [classification_rate;rate];
    
    if display_results
        fprintf('For K=%f, the classification rate is: %f\n',K, rate);
    end


    if size(various_K,2)==1
        % display complete test results (for debugging)
        figure;
        for batch = 1:10
            clf;
            for i = 1:12
                test_ind = (batch-1)*12+i;
                subplot(4,12,i);
                imshow(reshape(test_data(test_ind,:),[32 32]),[]);
                if i == 6
                    title('Orig. test img.');
                end
                subplot(4,12,i+12);
                imshow(reshape(test_data_reconstructed(test_ind,:),[32 32]),[]);
                if i == 6
                    title('Low-dim test img.');
                end
                subplot(4,12,i+24);
                imshow(reshape(train_data_reconstructed(nn_ind(test_ind),:),[32 32]),[]);
                if i == 6
                    title('Low-dim nearest neighbor');
                end
                subplot(4,12,i+36);
                imshow(reshape(train_data(nn_ind(test_ind),:),[32 32]),[]);
                if i == 6
                    title('Orig. nearest neighbor');
                end
                if estimated_label(test_ind)~=test_label(test_ind)
                    xlabel('incorrect');
                end
            end
            pause;
        end
    end


end % End for K values
toc;

if size(various_K,2)>1
    figure;
    plot(various_K,classification_rate);
    title('Classification rate');
    xlabel('Dimension K');
end