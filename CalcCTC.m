function [Weight,CorrectRate] = CalcCTC(NN,NNInput,charFeature,usedNNOutputs)
    NNOutput = NN.forward(NNInput);
    NNOutput(:,:,1:2  ,:) = softmax(NNOutput(:,:,1:2  ,:));% 1>2表示有声 否则表示没声（间断）
    NNOutput(:,:,3:26 ,:) = softmax(NNOutput(:,:,3:26 ,:));% 声母
    NNOutput(:,:,27:61,:) = softmax(NNOutput(:,:,27:61,:));% 韵母
    NNOutput(:,:,62:66,:) = softmax(NNOutput(:,:,62:66,:));% 音调
    % 概率计算负对数，方便后面用加法进行条件概率计算
    epsilon=1e-5;
    NNOutput(NNOutput < epsilon)=NNOutput(NNOutput < epsilon)+epsilon;
    % CTC一堆标量操作 用gpu算更慢 这里把数据取到CPU算
    NNOutput=gather(extractdata(-log(NNOutput)));
    Weight = zeros(size(NNOutput));
    Loss = 0;
    for i=1:size(NNInput,4)
        % 计算CTC得到序列映射关系矩阵
        [loss,Weight(:,1:usedNNOutputs(i),:,i)] = CTCLoss(NNOutput(:,1:usedNNOutputs(i),:,i), charFeature{i});
    Loss=Loss+loss;
    end
    
    % 统计正确率
    % softmax输出转换成编号
    % 前面取了负对数所以这里要最小值
    NNCategorical = zeros([size(NNOutput,[1 2]) ,4,size(NNOutput,4)]  );
    [~,NNCategorical(:,:,1,:)] = min(NNOutput(:,:,1:2,:),[],3);
    [~,NNCategorical(:,:,2,:)] = min(NNOutput(:,:,3:26,:),[],3);
    [~,NNCategorical(:,:,3,:)] = min(NNOutput(:,:,27:61,:),[],3);
    [~,NNCategorical(:,:,3,:)] = min(NNOutput(:,:,62:66,:),[],3);
    % CTC得到的匹配矩阵转换成编号
    WeightCategorical = zeros([size(Weight,[1 2]) ,4,size(Weight,4)]  );
    [~,WeightCategorical(:,:,1,:)] = max(Weight(:,:,1:2,:),[],3);
    [~,WeightCategorical(:,:,2,:)] = max(Weight(:,:,3:26,:),[],3);
    [~,WeightCategorical(:,:,3,:)] = max(Weight(:,:,27:61,:),[],3);
    [~,WeightCategorical(:,:,3,:)] = max(Weight(:,:,62:66,:),[],3);
    
    num=0;denum=0;
    for i=1:size(Weight,4)
        % 识别出来是静音（WeightCategorical（:,:,1,:)==2)的段不参与统计
        NNCategorical_Used = WeightCategorical(1,1:usedNNOutputs(i),1,i) == 1;
        % 多少个字识别对了(有无，声母，韵母，音调都识别对）
        num = num+sum(all(NNCategorical(1,NNCategorical_Used,:,i) == WeightCategorical(1,NNCategorical_Used,:,i),3),'all');
        % 一共有多少个字
        denum = denum+sum(NNCategorical_Used);
    end
    CorrectRate = 100*num/denum;
end