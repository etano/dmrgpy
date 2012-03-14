import pstats
p = pstats.Stats('dmrg.prof')
p.sort_stats('cumulative').print_stats(100)
