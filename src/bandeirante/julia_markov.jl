using Distributions, Random, LinearAlgebra

function initialize_parameters(s)
    A = rand(s,s)
    A = A./sum(A,dims=2) #check dim later

    P_0 = rand(s)
    P_0 = P_0./sum(P_0,dims=1) #check dim later

    return A, P_0
end

function initialize_gaussian(x,s)
    μ = rand(1,s)
    σ = rand(1,s)
    
    vector_of_distribs = [Normal(μ[i],σ[i]) for i = 1:s]
    
    pdf_on_x(obs) = pdf.(vector_of_distribs,obs)
    B = stack(pdf_on_x.(x)) 
    return μ , σ, B
end


function forward_step(A,B,P_0,n,s)
    alpha = zeros(n,s)
    scale = zeros(n)

    alpha[1,:] = P_0.*B[1]
    scale[1] = sum(alpha[1,:]) 
    alpha[1,:] /= scale[1]


    for t = 2:n
        alpha[t,:] = (A*alpha[t-1,:]).*B[:,t]
        scale[t] = sum(alpha[t,:]) 
        alpha[t,:] = alpha[t,:]/scale[t]
    end
    return alpha, scale
end

function backward_step(A,B,n,s,scale)
    beta = zeros(n,s);
    beta[n,:] .= 1;
    for t=n-1:-1:1
        beta[t,:] = (A*(B[:,t+1].*beta[t+1,:]))/scale[t]
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

function maximize_transition(alpha,beta,gamma,A,B,n)

    A_new = zeros(s,s)

    for t = 1:n-1
        Γ = (alpha[t,:]*beta[t+1,:]').*B[:,t+1]'.*A #(A*alpha[t,:])*(B[:,t+1].*beta[t+1,:])'
        Γ = Γ/(alpha[t,:]'*beta[t,:])
        A_new += (Γ./sum(Γ,dims=2))/n
    end

    P_0new = gamma[1,:]

    θ = norm(A_new-A,2)+norm(P_0new-P_0,2)

    return A_new, P_0new, θ

end

function maximize_gaussian(gamma,μ,σ)

    μ_new = sum((gamma.*x),dims=1)./sum(gamma,dims=1)

    σ_new = sum(gamma.*((x.-μ).^2),dims=1)./sum(gamma,dims=1)
   
    θ = norm(μ_new-μ,2)+norm(σ_new-σ,2)

    return μ_new, σ_new, θ
end


function EM_algorithm(x,n,s,ϵ = 1e-6)

    past_likehood = Inf
    δ = Inf

    A,P_0 = initialize_parameters(s)
    μ,σ,B = initialize_gaussian(x,s)


    while  δ > ϵ

        alpha,scale = forward_step(A,B,P_0,n,s)
        beta = backward_step(A,B,n,s,scale)

        gamma = gamma_calculation(alpha,beta,n,s)

        A,P_0 = maximize_transition(alpha,beta,gamma,A,B,n)
        μ,σ = maximize_gaussian(gamma,μ,σ)
        
        likehood = log(sum(scale))

        δ = abs(likehood-past_likehood)

        past_likehood = likehood
    end

    return A, P_0, μ, σ

end