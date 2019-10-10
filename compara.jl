using CSV, DataFrames, Plots, SolverBenchmark
pyplot(size=(600,400))

include("modelo.jl")
include("metodos.jl")
include("problemas.jl")

# incluir os métodos individuais aqui

function compara()
  # Colocar aqui a lista de métodos
  metodos = [newton, secante, bisseccao]

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
