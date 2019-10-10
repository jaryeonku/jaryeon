import ForwardDiff.derivative

mutable struct EquacaoNL{T <: Real}
  f           # função
  x₀ :: T     # Ponto inicial
  contador
  nome
end

EquacaoNL(f, x, nome = "Sem nome") = EquacaoNL(f, x, 0, nome)

ponto_inicial(enl :: EquacaoNL) = enl.x₀
contador(enl :: EquacaoNL) = enl.contador
function reset!(enl :: EquacaoNL)
  enl.contador = 0
end

function fun_val(enl :: EquacaoNL, x)
  enl.contador += 1
  return enl.f(x)
end

function der_val(enl :: EquacaoNL, x)
  enl.contador += 2
  return derivative(enl.f, x)
end

function seg_der_val(enl :: EquacaoNL, x)
  enl.contador += 4
  return derivative(y -> derivative(enl.f, y), x)
end
