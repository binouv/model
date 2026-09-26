# G13 pre-training data amendment

The first stage-0 data build at range 0..31 exhausted the remaining canonical space after denying G5S, G7R, G11R and G12 groups. No optimizer update and no held score existed. Before successful data generation, stage0 is expanded to 0..63. Stages1-3, model, arms, schedule seeds, steps, batch, objective, gates and fixed-endpoint selection remain unchanged.
