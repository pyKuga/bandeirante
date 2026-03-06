using Distributions, Random, LinearAlgebra

function initialize_parameters(s)
    A = rand(s,s)
    A = A./sum(A,dims=2) #check dim later

    P_0 = rand(s)
    P_0 = P_0./sum(P_0,dims=1) #check dim later

    return A, P_0
end

function initialize_gaussian(s)
    μ = rand(s)
    σ = rand(s)
    
    vector_of_distribs = [Normal(μ[i],σ[i]) for i = 1:s]
    
    return μ , σ, vector_of_distribs
end


function forward_step(A,B,P_0,n,s)
    alpha = zeros(n,s)
    scale = zeros(n)

    alpha[1,:] = P_0.*B[1,:]
    scale[1] = sum(alpha[1,:]) 
    alpha[1,:] /= scale[1]


    for t = 2:n
        alpha[t,:] = (A'*alpha[t-1,:]).*B[t,:]
        scale[t] = sum(alpha[t,:]) 
        alpha[t,:] = alpha[t,:]/scale[t]
    end
    return alpha, scale
end

function backward_step(A,B,n,s,scale)
    beta = zeros(n,s);
    beta[n,:] .= 1;
    for t=n-1:-1:1
        beta[t,:] = (A*(B[t+1,:].*beta[t+1,:]))/scale[t+1]
    end
    
    return beta
end

function gamma_calculation(alpha,beta,n,s)

    gamma = zeros(n,s)
    for t = 1:n
        gamma[t,:] = (alpha[t,:].*beta[t,:])/(alpha[t,:]'*beta[t,:])
    end
    return gamma

end

# function maximize_transition(alpha,beta,gamma,A,B,n)

#     A_new = zeros(s,s)

#     for t = 1:n-1
#         Γ = (alpha[t,:]*(beta[t+1,:].*B[t+1,:])').*A
#         Γ = Γ/(alpha[t,:]'*beta[t+1,:])
#         A_new += Γ.*gamma[t,:]
#     end   

#     A_new = A_new./sum(gamma,dims=1)
#     P_0new = gamma[1,:]

#     return A_new, P_0new

# end

function maximize_transition(alpha,beta,gamma,A,B,n)

    s = size(A,1)
    A_new = zeros(s,s)

    for t = 1:n-1
        tmp = beta[t+1,:] .* B[t+1,:]
        Γ = (alpha[t,:] * tmp') .* A
        Γ /= sum(Γ)
        A_new += Γ
    end

    den = sum(gamma[1:n-1,:], dims=1)
    den = max.(den, 1e-12)

    A_new = A_new ./ den'

    P_0new = gamma[1,:]

    return A_new, P_0new
end

function maximize_gaussian(x,gamma)

    μ_new = sum((gamma.*x),dims=1)./sum(gamma,dims=1)

    var = sum(gamma.*((x.-μ_new).^2),dims=1)./sum(gamma,dims=1)
    var = max.(var,1e-6)

    return μ_new, sqrt.(var)
end


function EM_algorithm(x,s,ϵ = 1e-4)

    n, = size(x)

    past_likehood = -Inf
    δ = Inf

    A,P_0 = initialize_parameters(s)

    μ,σ,vector_of_distribs = initialize_gaussian(s)

    while  δ > ϵ

        B_func(x) = stack([pdf.(dist,x) for dist in vector_of_distribs])
        B = B_func(x)

        alpha,scale = forward_step(A,B,P_0,n,s)
        beta = backward_step(A,B,n,s,scale)

        gamma = gamma_calculation(alpha,beta,n,s)

        A,P_0 = maximize_transition(alpha,beta,gamma,A,B,n)
        μ,σ = maximize_gaussian(x,gamma,μ,σ)


        vector_of_distribs = [Normal(μ[i],σ[i]) for i = 1:s]

        
        likehood = sum(log.(scale))

        δ = likehood-past_likehood

        if δ < 0
            error("Likelihood decreased — bug numérico")
        end

        past_likehood = likehood
    end

    B_func(x) = stack([pdf.(dist,x) for dist in vector_of_distribs])

    return A, B_func, P_0, μ, σ, scale

end

function viterbi(x,A,B,P_0)
    n, = size(x)
    s, = size(A)

    B_vec = B(x)

    state_probs = zeros(n,s)   

    state_probs[1,:] = P_0.*B_vec[1,:]

    for i=2:n
        state_probs[i,:] = state_probs[i-1,:].*(A*B_vec[i,:])
    end

    states = argmax.(eachrow(state_probs))

    return states, state_probs        

end

function simulate_hmm(A, P_0, μ, σ, T)

    s = length(π)

    z = zeros(Int, T)
    x = zeros(T)

    # estado inicial
    z[1] = rand(Categorical(π))

    # emissão inicial
    x[1] = rand(Normal(μ[z[1]], σ[z[1]]))

    for t in 2:T
        # transição
        z[t] = rand(Categorical(A[z[t-1], :]))

        # emissão
        x[t] = rand(Normal(μ[z[t]], σ[z[t]]))
    end

    return z, x
end