# G13R runtime note

Before any completed endpoint or held score, CPU execution was changed to flush subnormal floating-point values to zero. Registered data, batch order, parameterization, objective, optimizer schedule, endpoint, parser, and gates are unchanged. Two earlier incomplete local attempts produced no durable checkpoint and are excluded from results.
