
function newton(enl :: EquacaoNL{T};
                atol = √eps(T),
                rtol = √eps(T),
                max_iter = 10_000,
                max_time = 30.0,
                max_eval = 10_000,
               ) where T <: AbstractFloat

  f(x) = fun_val(enl, x)
  g(x) = der_val(enl, x)
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
    if abs(gx) < atol
      status = :falha
      break
    end
    x -= fx / gx
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

function secante(enl :: EquacaoNL{T};
                 atol = √eps(T),
                 rtol = √eps(T),
                 max_iter = 10_000,
                 max_time = 30.0,
                 max_eval = 10_000,
                ) where T <: AbstractFloat

  f(x) = fun_val(enl, x)
  x = enl.x₀
  fx = f(x)
  xold = x + 1
  fold = f(xold)
  if fold < fx
    x, xold, fx, fold = xold, x, fold, fx
  end

  ϵ = atol + rtol * abs(fx)
  resolvido = abs(fx) ≤ ϵ
  start_time = time()
  Δt = 0.0
  iter = 0
  excedido = Δt > max_time || iter ≥ max_iter || contador(enl) ≥ max_eval

  status = :desconhecido
  while !(resolvido || excedido)
    gx = (fold - fx) / (xold - x)
    if abs(gx) < atol
      status = :falha
      break
    end
    xold, fold = x, fx
    x -= fx / gx
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

function bisseccao(enl :: EquacaoNL{T};
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

  x = (a + b) / 2
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
    x = (a + b) / 2
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
