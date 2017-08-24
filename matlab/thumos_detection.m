threshold = 0.5; T = 3;
load('~/code/temporal-segment-networks/THUMOS14_evalkit_20150930/test_score.mat');
%% Aggregation
flow_final_score = zeros(length(flow_test_score), 101);
for i = 1:length(flow_test_score)
    flow_final_score(i,:) = aggregation_attention(flow_test_score{i}, flow_test_attention{i}, 3);
end
flow_softmax_score = softmax(flow_final_score, 2);

rgb_final_score = zeros(length(rgb_test_score), 101);
for i = 1:length(rgb_test_score)
    rgb_final_score(i,:) = aggregation_attention(rgb_test_score{i}, rgb_test_attention{i}, 3);
end
rgb_softmax_score = softmax(rgb_final_score, 2);
fusion_softmax_score = (flow_softmax_score + rgb_softmax_score)/2 ;
[rec_all,prec_all,ap_all,map] = TH14evalRecog(fusion_softmax_score,'test_set_final.mat');


%% Setup
gtpath = 'annotation'; subset='test';
[th14classids,th14classnames]=textread([gtpath '/detclasslist.txt'],'%d%s');
th14classnamesamb = th14classnames;
th14classnamesamb=cat(1,th14classnames,'Ambiguous');
clear gtevents
gteventscount=0;
for i=1:length(th14classnamesamb)
  class=th14classnamesamb{i};
  gtfilename=[gtpath '/' class '_' subset '.txt'];
  if exist(gtfilename,'file')~=2
    error(['TH14evaldet: Could not find GT file ' gtfilename])
  end
  [videonames,t1,t2]=textread(gtfilename,'%s%f%f');
  for j=1:length(videonames)
    gteventscount=gteventscount+1;
    gtevents(gteventscount).videoname=videonames{j};
    gtevents(gteventscount).timeinterval=[t1(j) t2(j)];
    gtevents(gteventscount).class=class;
    gtevents(gteventscount).conf=1;
  end
end
gtvideonames={gtevents.videoname};
videoname = unique(gtvideonames);

idx = zeros(1, length(videoname));
for i = 1:length(idx)
    idx(i) = str2num(videoname{i}(end-4:end));
end


%% detection
testing_id = idx;
fid = fopen(['tmp', '.txt'],'w');
gtpath= 'annotation/';
load('test_set_final.mat');
[th14classids,th14classnames]=textread([gtpath '/detclasslist.txt'],'%d%s');

for i = 1:length(th14classids)
    for jj = 1:length(testing_id)
        j = testing_id(jj);
        attention = rgb_test_attention{j};
        score = rgb_test_score{j};
        
        attention1 = flow_test_attention{j};
        score1 = flow_test_score{j};
        if size(attention1, 1) < size(attention, 1)
            attention1 = cat(1,attention1, attention1(end));
        end
        if size(score1, 1) < size(score, 1)
            score1 = cat(1, score1, score1(end,:));
        end
        
        video = test_videos(j).video_name;
        fps = test_videos(j).frame_rate_FPS;
        duration = test_videos(j).video_duration_seconds;
        
        frame_score = softmax(score, 2);
        frame_score = frame_score(:, th14classids(i));
        
        frame_score1 = softmax(score1, 2);
        frame_score1 = frame_score1(:, th14classids(i));
        
        f_score = (frame_score + frame_score1)/2;
        f_score = score_smooth(f_score);
        
        c_score= fusion_softmax_score(j, th14classids(i));
        
        if c_score > 0.1
            mask =  (f_score > max(f_score)*threshold);
            mask = imdilate(mask, [1,1]);
            L = bwlabeln(mask, 4);
            for k = 1:max(L)
                idx = L==k;
                tmp = f_score(idx);
                det_score = mean(tmp) + 0.2*c_score;
                start = find(L==k, 1 );
                start = 1/fps + (start-1)*15/fps;
                last = find(L==k, 1, 'last' );
                last = 1/fps + (last)*15/fps;
                fprintf(fid, '%s %d %d %d %d\n', video, start, last, th14classids(i), det_score);
            end
        end
        
    end
end
fclose(fid);

[P, A, MAP] = TH14evalDet('tmp.txt', 'annotation', 'test', 0.1);
