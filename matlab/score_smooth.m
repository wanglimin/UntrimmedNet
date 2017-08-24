function smooth = score_smooth(score)
score = score';
tmp1 = [score(2:end),score(end)];
tmp2 = [score(1), score(1:end-1)];
smooth = (score +  tmp1 + 0*tmp2)/2;
end