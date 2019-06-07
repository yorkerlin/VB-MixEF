function [grad,g_mean,g_var] = adam(gradient,g_mean,g_var,decay_factor_mean,decay_factor_var,epsilon,learning_rate,times)
        g_mean=decay_factor_mean .* g_mean + (1.0-decay_factor_mean) .* (gradient);
        g_var=decay_factor_var .* g_var + (1.0-decay_factor_var) .* (gradient.^2);
        g_mean_hat=g_mean ./ (1.0-(decay_factor_mean.^times));
        g_var_hat=g_var ./ (1.0-(decay_factor_var.^times));
        grad=learning_rate .* g_mean_hat ./ (sqrt(g_var_hat)+epsilon);
end

