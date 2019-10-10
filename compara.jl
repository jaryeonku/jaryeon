using CSV, DataFrames, Plots, SolverBenchmark
pyplot(size=(600,400))

include("modelo.jl")
include("metodos.jl")
include("problemas.jl")

function newtonadaptado(enl :: EquacaoNL{T};
                atol = √eps(T),
                rtol = √eps(T),
                max_iter = 10_000,
                max_time = 30.0,
                max_eval = 10_000,
               ) where T <: AbstractFloat

  f(x) = fun_val(enl, x)
  g(x) = der_val(enl, x)
  h(x) = seg_der_val(enl, x) 
  x = enl.x₀
  fx = f(x)
  ϵ = atol + rtol * abs(fx)

  resolvido = abs(fx) ≤ ϵ
  start_time = time()
  Δt = 0.0
  iter = 0
  excedido = Δt > max_time || iter ≥ max_iter || contador(enl) ≥ max_eval

  status = :desconhecido
  while !(resolvido || excedido)
    gx = g(x)
    hx = h(x)
    if abs(gx) < atol
      status = :falha
      break
    end
    x -= fx * gx / ( gx^2 - fx * hx)
    fx = f(x)
    resolvido = abs(fx) ≤ ϵ
    Δt = time() - start_time
    iter += 1
    excedido = Δt > max_time || iter ≥ max_iter || contador(enl) ≥ max_eval
  end

  if resolvido
    status = :resolvido
  elseif excedido
    if Δt > max_time
      status = :max_time
    else
      status = :max_iter
    end
  end

  return x, fx, status, iter, Δt
end


function falsaposicao(enl :: EquacaoNL{T};
                   atol = √eps(T),
                   rtol = √eps(T),
                   max_iter = 10_000,
                   max_time = 30.0,
                   max_eval = 10_000,
                  ) where T <: AbstractFloat

  start_time = time()
  f(x) = fun_val(enl, x)
  a = enl.x₀
  fa = f(a)
  ϵ = atol + rtol * abs(fa)
  if abs(fa) ≤ ϵ
    return a, fa, :resolvido, 0, time() - start_time
  end
  b = a + 1
  fb = f(b)
  if abs(fb) ≤ ϵ
    return b, fb, :resolvido, 0, time() - start_time
  end

  while fa * fb ≥ 0
    δ = b - a
    if δ > 1e5
      return a, fa, :falha, 0, time() - start_time
    end
    if abs(fa) < abs(fb)
      a -= δ
      fa = f(a)
    else
      b += δ
      fb = f(b)
    end
  end

  x = (a*fb + b*fa) / (fb - fa)
  fx = f(x)
  resolvido = abs(fx) ≤ ϵ
  Δt = time() - start_time
  iter = 0
  excedido = Δt > max_time || iter ≥ max_iter || contador(enl) ≥ max_eval

  while !(resolvido || excedido)
    if fa * fx < 0
      b, fb = x, fx
    else
      a, fa = x, fx
    end
    x = (a*fb + b*fa) / (fb - fa)
    fx = f(x)
    resolvido = abs(fx) ≤ ϵ
    iter += 1
    Δt = time() - start_time
    excedido = Δt > max_time || iter ≥ max_iter || contador(enl) ≥ max_eval
  end

  status = :desconhecido
  if resolvido
    status = :resolvido
  elseif excedido
    if Δt > max_time
      status = :max_time
    else
      status = :max_iter
    end
  end

  return x, fx, status, iter, Δt
end

function compara()
  # Colocar aqui a lista de métodos
  metodos = [newton, secante, bisseccao, newtonadaptado, falsaposicao]

  data = Dict{Symbol,DataFrame}()
  for mtd in metodos
    df = DataFrame(:id => Int[],
                   :name => String[],
                   :status => Symbol[],
                   :x => Real[],
                   :fx => Real[],
                   :iter => Int[],
                   :time => Float64[],
                   :eval => Int[],
                  )
    for (id, p) in enumerate(ProblemList)
      reset!(p)
      x, fx, status, iter, Δt = mtd(p)
      push!(df, [id, p.nome, status, x, fx, iter, Δt, contador(p)])
    end
    data[Symbol(mtd)] = df
    CSV.write("$mtd.csv", df)
  end

  # Exemplo de comparação usando SolverBenchmark
  cost(df) = begin
    V = df.eval
    I = findall(df.status .!= :resolvido)
    V[I] .= -1
    return V
  end
  performance_profile(data, cost)
  png("perfil")

  return data
end

data = compara()
