function softmax_score = softmax(final_score, softmax_axis)

softmax_score = exp(final_score);
softmax_score = bsxfun(@rdivide, softmax_score, sum(softmax_score, softmax_axis));

end