function [Loss,Grad] = NN_Run(NN,NNInput,Weight)
    % 数据送进神经网络
    NNOutput = NN.forward(NNInput);
    % matlab自己的dlnetwork不能接好几个softmax层输出 所以这里自己算softmax
    NNOutput(:,:,1:2  ,:) = softmax(NNOutput(:,:,1:2  ,:));
    NNOutput(:,:,3:26 ,:) = softmax(NNOutput(:,:,3:26 ,:));
    NNOutput(:,:,27:61,:) = softmax(NNOutput(:,:,27:61,:));
    NNOutput(:,:,62:66,:) = softmax(NNOutput(:,:,62:66,:));
    % 这里取各字的概率负对数和做loss
    epsilon=1e-5;
    NNOutput(NNOutput < epsilon)=NNOutput(NNOutput < epsilon)+epsilon;
    % CTC不能放自动求导里，否则巨慢，这里在外面算好Weight在送进dlfeval
    Loss=sum(-log(NNOutput).*Weight,'all');
    Loss=log(1+Loss);
    Grad = dlgradient(Loss,NN.Learnables);
end