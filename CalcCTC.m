function [Weight,CorrectRate] = CalcCTC(NN,NNInput,charFeature,usedNNOutputs)
    NNOutput = NN.forward(NNInput);
    NNOutput(:,:,1:2  ,:) = softmax(NNOutput(:,:,1:2  ,:));% 1>2��ʾ���� �����ʾû������ϣ�
    NNOutput(:,:,3:26 ,:) = softmax(NNOutput(:,:,3:26 ,:));% ��ĸ
    NNOutput(:,:,27:61,:) = softmax(NNOutput(:,:,27:61,:));% ��ĸ
    NNOutput(:,:,62:66,:) = softmax(NNOutput(:,:,62:66,:));% ����
    % ���ʼ��㸺��������������üӷ������������ʼ���
    epsilon=1e-5;
    NNOutput(NNOutput < epsilon)=NNOutput(NNOutput < epsilon)+epsilon;
    % CTCһ�ѱ������� ��gpu����� ���������ȡ��CPU��
    NNOutput=gather(extractdata(-log(NNOutput)));
    Weight = zeros(size(NNOutput));
    Loss = 0;
    for i=1:size(NNInput,4)
        % ����CTC�õ�����ӳ���ϵ����
        [loss,Weight(:,1:usedNNOutputs(i),:,i)] = CTCLoss(NNOutput(:,1:usedNNOutputs(i),:,i), charFeature{i});
    Loss=Loss+loss;
    end
    
    % ͳ����ȷ��
    % softmax���ת���ɱ��
    % ǰ��ȡ�˸�������������Ҫ��Сֵ
    NNCategorical = zeros([size(NNOutput,[1 2]) ,4,size(NNOutput,4)]  );
    [~,NNCategorical(:,:,1,:)] = min(NNOutput(:,:,1:2,:),[],3);
    [~,NNCategorical(:,:,2,:)] = min(NNOutput(:,:,3:26,:),[],3);
    [~,NNCategorical(:,:,3,:)] = min(NNOutput(:,:,27:61,:),[],3);
    [~,NNCategorical(:,:,3,:)] = min(NNOutput(:,:,62:66,:),[],3);
    % CTC�õ���ƥ�����ת���ɱ��
    WeightCategorical = zeros([size(Weight,[1 2]) ,4,size(Weight,4)]  );
    [~,WeightCategorical(:,:,1,:)] = max(Weight(:,:,1:2,:),[],3);
    [~,WeightCategorical(:,:,2,:)] = max(Weight(:,:,3:26,:),[],3);
    [~,WeightCategorical(:,:,3,:)] = max(Weight(:,:,27:61,:),[],3);
    [~,WeightCategorical(:,:,3,:)] = max(Weight(:,:,62:66,:),[],3);
    
    num=0;denum=0;
    for i=1:size(Weight,4)
        % ʶ������Ǿ�����WeightCategorical��:,:,1,:)==2)�Ķβ�����ͳ��
        NNCategorical_Used = WeightCategorical(1,1:usedNNOutputs(i),1,i) == 1;
        % ���ٸ���ʶ�����(���ޣ���ĸ����ĸ��������ʶ��ԣ�
        num = num+sum(all(NNCategorical(1,NNCategorical_Used,:,i) == WeightCategorical(1,NNCategorical_Used,:,i),3),'all');
        % һ���ж��ٸ���
        denum = denum+sum(NNCategorical_Used);
    end
    CorrectRate = 100*num/denum;
end