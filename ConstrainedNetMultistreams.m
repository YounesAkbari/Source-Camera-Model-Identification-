% % 
% digitDatasetPathtrain = fullfile('C:\Users\VRLab\Desktop\Younes\ForgeryDatabase\PatchForEvaluationGray\Training');
% digitDatasetPathtest = fullfile('C:\Users\VRLab\Desktop\Younes\ForgeryDatabase\PatchForEvaluationGray\Testing');
% digitDatasetPathvalidation = fullfile('C:\Users\VRLab\Desktop\Younes\ForgeryDatabase\PatchForEvaluationGray\Validation');
% imds = imageDatastore(digitDatasetPathtrain, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% imdstest = imageDatastore(digitDatasetPathtest, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% imdsvalidation = imageDatastore(digitDatasetPathvalidation, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% % numTrainingFiles =600000;
%numTrainingFiles =30000;
%[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');
% % %[imdsValid,imdsTest1] = splitEachLabel(imdsTest,numTrainingFiles1,'randomize');
% % %numTestingFiles = 900;
%     imageSize = [350 350 1];
%     augimdsvalid = augmentedImageDatastore(imageSize,imdsvalidation,'OutputSizeMode','centercrop');
  % augimdstrain = augmentedImageDatastore(imageSize,imds,'OutputSizeMode','centercrop');
% %[imdsTest2,imdsValid] = splitEachLabel(imdsTest,numTestingFiles,'randomize');
% 
% 
% %augimdsvalid = augmentedImageDatastore(imageSize,imdsvalidation,'OutputSizeMode','centercrop');
% %augimdstest = augmentedImageDatastore(imageSize,imdstest,'OutputSizeMode','centercrop');
% %numberOfWorkers=4;
 miniBatchSize =64;
%  initialLearnRate = 1e-1 * miniBatchSize/256;'WorkerLoad', [0 1 0 1],
% % % % % %  'Plots','training-progress', ... % Turn on the training progress plot.

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu', ... % Turn on automatic parallel support.
    'MiniBatchSize',miniBatchSize, ... % Set the MiniBatchSize.
    'Verbose',false, ... % Do not send command line output.
    'L2Regularization',1e-5, ...
    'MaxEpochs',15, ...
    'Plots','training-progress', ... % Turn on the training progress plot.
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsvalid, ...
    'ValidationFrequency',floor(numel(augimdsvalid.Files)/miniBatchSize), ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',45);


%augimds = augmentedImageDatastore(imageSize,imds,'OutputSizeMode','randcrop');


%lgraph = addLayers(lgraph,FC1);
% lgraph = addLayers(lgraph,Tan11);
% lgraph = connectLayers(lgraph,'conv_Constrain1','conv_20');
% lgraph = connectLayers(lgraph,'conv_20','BN_10');
% 
% concat = concatenationLayer(2,2,'Name','concat');
% lgraph = addLayers(lgraph, concat);,'Normalization','none'
lgraph = layerGraph;
layers1 = [ 
  imageInputLayer([350 350 1],'Name','Layer1_Input','Normalization','none')
  
    %wavePooling
     %convolution2dLayer([5 5],3,'Name','conv_1','Padding','same')
     convolution2dLayer([5 5],3,'Name','Layer1_conv_Constrain')
    convolution2dLayer([7 7],34,'Name','Layer1_conv_1')
    batchNormalizationLayer('Name','Layer1_BN1')
    tanhLayer('Name','Layer1_tan1')
    
    maxPooling2dLayer(3,'Stride',2,'Name','Layer1_maxPool1')
    
    convolution2dLayer([5 5],24,'Name','Layer1_conv2')
    batchNormalizationLayer('Name','Layer1_BN2')
    tanhLayer('Name','Layer1_tan2')
    
    maxPooling2dLayer(3,'Stride',2,'Name','Layer1_maxPool2')
    
    convolution2dLayer([5 5],24,'Name','Layer1_conv3')
    batchNormalizationLayer('Name','Layer1_BN3')
    tanhLayer('Name','Layer1_tan3')
    
    maxPooling2dLayer(3,'Stride',2,'Name','Layer1_maxPool3')
    
    
    convolution2dLayer([1 1],34,'Name','Layer1_conv4')
    batchNormalizationLayer('Name','Layer1_BN4')
    tanhLayer('Name','Layer1_tan4')
    %wavePooling
    maxPooling2dLayer(3,'Stride',2,'Name','Layer1_maxPool4')
    
    fullyConnectedLayer(512,'Name','Layer1_FC1')
    tanhLayer('Name','Layer1_tan5')];





layers2 = [ 
    
     fullyConnectedLayer(1024,'Name','Layer2_FC1')
    tanhLayer('Name','Layer2_tan2')
    fullyConnectedLayer(512,'Name','Layer2_FC2')
    tanhLayer('Name','Layer2_tan3')
    
%     fullyConnectedLayer(200,'Name','Layer2_FC3')
%     tanhLayer('Name','Layer4_tan4')
%     fullyConnectedLayer(100,'Name','Layer2_FC4')
%     tanhLayer('Name','Layer2_tan5')
    
    fullyConnectedLayer(10,'Name','Layer2_FC5')
    softmaxLayer('Name','SM')
    classificationLayer('Name','Classi')];

layers3 = [ 
    batchNormalizationLayer('Name','Layer3_BN1')
    tanhLayer('Name','Layer3_tan1')
    maxPooling2dLayer(3,'Stride',2,'Name','Layer3_maxPool1')
    maxPooling2dLayer(3,'Stride',2,'Name','Layer3_maxPool2')
    fullyConnectedLayer(1024,'Name','Layer3_FC1')
    tanhLayer('Name','Layer3_tan2')
%     fullyConnectedLayer(2048,'Name','Layer3_FC2')
%     tanhLayer('Name','Layer3_tan3')
    
    fullyConnectedLayer(512,'Name','Layer3_FC2')
%     tanhLayer('Name','Layer3_tan4')
%     fullyConnectedLayer(512,'Name','Layer3_FC4')
    tanhLayer('Name','Layer3_tan5')];

layers4 = [ 
    
    fullyConnectedLayer(512,'Name','Layer4_FC1')
    tanhLayer('Name','Layer4_tan2')
%     fullyConnectedLayer(1024,'Name','Layer4_FC2')
%     tanhLayer('Name','Layer4_tan3')
    
 ];

layers5 = [ 
    
    fullyConnectedLayer(512,'Name','Layer5_FC1')
    tanhLayer('Name','Layer5_tan2')
%     fullyConnectedLayer(1024,'Name','Layer4_FC2')
%     tanhLayer('Name','Layer4_tan3')
    
 ];
layers6 = [ 
    
    fullyConnectedLayer(512,'Name','Layer6_FC1')
    tanhLayer('Name','Layer6_tan2')
%     fullyConnectedLayer(1024,'Name','Layer4_FC2')
%     tanhLayer('Name','Layer4_tan3')
    
 ];

lgraph = addLayers(lgraph,layers1);
lgraph = addLayers(lgraph,layers2);
lgraph = addLayers(lgraph,layers3);
lgraph = addLayers(lgraph,layers4);
lgraph = addLayers(lgraph,layers5);
lgraph = addLayers(lgraph,layers6);

Fusion = depthConcatenationLayer(5,'Name','Fusion');
%Fusion=additionLayer(5,'Name','Fusion');
lgraph = addLayers(lgraph, Fusion);

lgraph = connectLayers(lgraph,'Layer1_conv_Constrain','Layer3_BN1');
lgraph = connectLayers(lgraph,'Layer1_maxPool1','Layer4_FC1');
lgraph = connectLayers(lgraph,'Layer1_maxPool2','Layer5_FC1');
lgraph = connectLayers(lgraph,'Layer1_maxPool3','Layer6_FC1');
lgraph = connectLayers(lgraph, 'Layer1_tan5', 'Fusion/in1');
lgraph = connectLayers(lgraph, 'Layer3_tan5', 'Fusion/in2');
lgraph = connectLayers(lgraph, 'Layer4_tan2', 'Fusion/in3');
lgraph = connectLayers(lgraph, 'Layer5_tan2', 'Fusion/in4');
lgraph = connectLayers(lgraph, 'Layer6_tan2', 'Fusion/in5');
lgraph = connectLayers(lgraph,'Fusion','Layer2_FC1');
% parfor i=1:gpuDeviceCount("available")
%ile = gpuArray(0.0001);
 net = trainNetwork(imdsTrain,lgraph,options);
% end