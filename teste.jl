using Test

include("modelo.jl")
include("metodos.jl")

function testes()
  for mtd in [newton, secante, bisseccao]
    @testset "Testes para $mtd" begin
      for (f, x, s) in [(x -> 3x - 5, 1.0, 5 / 3),
                        (x -> x^2 - 2, 1.0, sqrt(2)),
                        (x -> x^2 - 2, big"1.0", sqrt(big"2.0")),
                        (x -> x * exp(x) - 1, 1.0, 0.567143290),
                        (x -> x^3 - 2^3, 1.0, 2.0),
                        (x -> x^4 - 2^4, 1.0, 2.0),
                        (x -> x^5 - 2^5, 1.0, 2.0),
                        (x -> sin(x) + max(3 - x, 0)^2 - max(x - 3.2, 0)^2, 1.0, π),
                        (x -> sin(x) + max(3 - x, 0)^2 - max(x - 3.2, 0)^2, big"1.0", π),
                        (x -> 1 / (1 + exp(-0.1x + 1)) - 0.5, 0.0, 10.0),
                       ]

        enl = EquacaoNL(f, x)
        xs, fx, status, iter, Δt = mtd(enl)
        @test isapprox(xs, s, atol=1e-5)
        @test status == :resolvido
      end
    end
  end
end

testes()
