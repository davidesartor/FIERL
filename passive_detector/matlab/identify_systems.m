for i = 1:100
    i
    table = patTS_Dataset{i};
    data=iddata(table.CGM, [table.insulin, table.CHO]);
    [data_dtr, tr] = detrend(data);
    model_Tf=armax(data_dtr,[7 7 7 7 6 3]);
    modelSS=idss(model_Tf);

    if i == 1
        A = modelSS.A;
        B = modelSS.B;
        C = modelSS.C;
        D = modelSS.D;
        K = modelSS.K;
        NoiseVar = modelSS.NoiseVariance;
        stable = isstable(modelSS);
        u_dtr = data_dtr.u;
        y_dtr = data_dtr.y;
    else
        A = cat(3, A, modelSS.A);
        B = cat(3, B, modelSS.B);
        C = cat(3, C, modelSS.C);
        D = cat(3, D, modelSS.D);
        K = cat(3, K, modelSS.K);
        NoiseVar = cat(2, NoiseVar, modelSS.NoiseVariance);
        stable = cat(2, stable, isstable(modelSS));
        u_dtr = cat(3, u_dtr, data_dtr.u);
        y_dtr = cat(3, y_dtr, data_dtr.y);
    end
end