function [rec_all,prec_all,ap_all,map]=TH14evalRecog(InputScore,gtpath)

% [rec_all,prec_all,ap_all,map]=TH14evalRecog(clsfilename,gtpath)
%
%     Input:    clsfilename: path of the input file
%                    gtpath: the path of the groundtruth file
%
%    Output:        rec_all: recall
%                  prec_all: precision
%                    ap_all: AP for each class
%                       map: MAP 
%
% Example:
%
%  [rec_all,prec_all,ap_all,map]=TH14evalRecog('Run-1.txt','test_set_final.mat');
%


load(gtpath);
ntest = size(test_videos,2);
gtlabel = zeros(ntest, 101);

for i=1:ntest
    videoName = regexprep(test_videos(i).video_name,'video_test_','');
    if strcmp(test_videos(i).background_video,'NO')
        gtlabel(str2num(videoName),test_videos(i).primary_action_index) = 1;
        if test_videos(i).secondary_actions_indices
            gtlabel(str2num(videoName),test_videos(i).secondary_actions_indices) = 1;
        end
    end
end

for i=1:101
    [rec_all(:,i),prec_all(:,i),ap_all(:,i)]=TH14eventclspr(InputScore(:,i),gtlabel(:,i));
end
map=mean(ap_all);
fprintf('\n\nMAP: %f \n\n',map);

% num = 0;
% [v, idx] = max(InputScore, [],2);
% for i = 1:size(InputScore, 1)
%     num = num + gtlabel(i, idx(i));
% end
% num/ntest

function [rec,prec,ap]=TH14eventclspr(conf,labels)
[so,sortind]=sort(-conf);
tp=labels(sortind)==1;
fp=labels(sortind)~=1;
npos=length(find(labels==1));

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision

ap=0;
tmp=labels(sortind)==1;
for i=1:length(conf)
    if tmp(i)==1
        ap=ap+prec(i);
    end
end
ap=ap/npos;


