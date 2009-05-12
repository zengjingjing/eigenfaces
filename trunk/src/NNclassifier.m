function [nn_ind, estimated_label] = NNclassifier(train_data_reconstructed,train_label,test_data_reconstructed);

nn_ind=[];
estimated_label=[];
distance=[];
for i=1:size(test_data_reconstructed,1)
    for j=1:size(train_data_reconstructed,1)
        distance(j) = corr2(test_data_reconstructed(i,:),train_data_reconstructed(j,:)); %Note: no need to reshape to 32*32 pic
%        distance(j) = corr2(reshape(test_data_reconstructed(i,:),[32,32]),reshape(train_data_reconstructed(j,:),[32,32])); 
    end
    [d_min,ind] = max(distance);
    estimated_label = [estimated_label;train_label(ind)];
    nn_ind = [nn_ind;ind];
end
