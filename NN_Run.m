function [Loss,Grad] = NN_Run(NN,NNInput,Weight)
    % �����ͽ�������
    NNOutput = NN.forward(NNInput);
    % matlab�Լ���dlnetwork���ܽӺü���softmax����� ���������Լ���softmax
    NNOutput(:,:,1:2  ,:) = softmax(NNOutput(:,:,1:2  ,:));
    NNOutput(:,:,3:26 ,:) = softmax(NNOutput(:,:,3:26 ,:));
    NNOutput(:,:,27:61,:) = softmax(NNOutput(:,:,27:61,:));
    NNOutput(:,:,62:66,:) = softmax(NNOutput(:,:,62:66,:));
    % ����ȡ���ֵĸ��ʸ���������loss
    epsilon=1e-5;
    NNOutput(NNOutput < epsilon)=NNOutput(NNOutput < epsilon)+epsilon;
    % CTC���ܷ��Զ��������������������������Weight���ͽ�dlfeval
    Loss=sum(-log(NNOutput).*Weight,'all');
    Loss=log(1+Loss);
    Grad = dlgradient(Loss,NN.Learnables);
end