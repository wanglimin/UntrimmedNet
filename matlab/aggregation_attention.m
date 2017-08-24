function final_score = aggregation_attention(score, attention, T)

if size(score, 1)
    score = cat(1, score, score);
    attention = cat(1, attention, attention);
end

if size(score, 3) > 1
score = double(squeeze(mean(score, 2)));
attention = double(squeeze(mean(attention, 2)));
end

attention = attention/T;
softmax_attention = exp(attention);
softmax_attention = bsxfun(@rdivide, softmax_attention, sum(softmax_attention));

final_score = bsxfun(@times, score, softmax_attention);
final_score = sum(final_score, 1);

end