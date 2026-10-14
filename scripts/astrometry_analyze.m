%%

RAD  = 180./pi;
Long = -70.75;
Lat  = -29.25;

% read the image names
List  = io.files.filelist('BG*.fits');
Nlist = numel(List);

Xgal = 6820;
Ygal = 2860;
HalfWidth = 500; %1000;

% BG pix scale [arcsec/pix]
PixScale = 0.5643;

% SN J2000 RA/Dec [deg]
RA  = 287.35745;
Dec = -63.83384;


IndF = 0; % index for found SN
AssymRMS = nan(Nlist,1);
Nastrom  = nan(Nlist,1);
JD       = nan(Nlist,1);
Filter   = cell(Nlist,1);
% go over all images
for Ilist=1:1:Nlist
    tic;
    AI = AstroImage(List{Ilist});
    
    % get SN pix position:
    [Xsn, Ysn] = AI.WCS.sky2xy(RA,Dec);

    % trim the images around the SN position
    %AI.HeaderData.deleteDistortionsWCS;
    AI = AI.crop([Xsn Ysn HalfWidth HalfWidth] , 'Type','center');
    % note that there is a bug in crop (WCS not updated) probabloy related
    % to the BG header. Will likely be solved by deleting the BG WCS
    % keywords.

    % The images looks that they are background subtracted - bad practice
    % Is there a mask image?
    % what is the GAIN? GAIN=1, but GAIN%d>1

    % add background
    AI = AI + 1000;

    % estimate background
    AI = imProc.background.background(AI); %, 'BackFun',@median, 'BackFunPar',{[1 2]}, 'VarFun',@imUtil.background.rvar);

    AI = imProc.sources.findMeasureSources(AI);
    % test quality of BG astrometry - really bad, with offsets of ~0.1-0.2"
    %AI = imProc.astrometry.addCoordinates2catalog(AI);
    % RQ=imProc.astrometry.astrometryQualityData(AI)

    Filter{Ilist} = AI.HeaderData.Key.FILTER;
    JD(Ilist)     = AI.julday;

    %AI.CatData.sortrows('Y1');
    UsePSFphot = true;
    if UsePSFphot
        AI = imProc.psf.populatePSF(AI);
        AI = imProc.sources.psfFitPhot(AI);
    end
    AI = imProc.astrometry.addCoordinates2catalog(AI,'OutUnits','deg');
    AI.CatData.sortrows('Dec');
    AI = imProc.calib.photometricZP(AI);

    try
        [ResAst, AI] = imProc.astrometry.astrometryCore(AI, 'RA',RA, 'Dec',Dec, 'Scale',PixScale);
    
        KeepImages = true;
        if KeepImages
            AllAI(Ilist) = AI.copy;
            % keep only AstroCatalog
            AC(Ilist) = AI.CatData;
        else
            % keep only AstroCatalog
            AC(Ilist) = AI.CatData;
        end
    
        % keep asymp. rms for each image
        AssymRMS(Ilist) = ResAst.ResFit.AssymRMS.*3600;
        Nastrom(Ilist)  = ResAst.ResFit.Ngood;
    
        [Ilist, AssymRMS(Ilist), toc]
    catch ME
        sprintf('Error wjhile analyzing image %d',Ilist);

    end



    % find SN
    %Dist = AI.CatData.sphere_dist(RA,Dec,'deg','arcsec');
    %[MinDist, MinInd] = min(Dist);
    %if MinDist<1
    %    % SN found keep data
    %    IndF = IndF+1;
    %end

    % inspect image
    %ds9(AI);
    %ds9.plot(AI)
    %ds9.plotc(RA,Dec)

end

%% Coadd

CurrFilter = 'u';
FlagGoodAstrometry = AssymRMS>0.001 & AssymRMS<0.04 & Nastrom>60 & strcmp(Filter,CurrFilter); % & abs(HA)<20;
sum(FlagGoodAstrometry)
II = find(FlagGoodAstrometry);

RegAI = imProc.transIm.interp2wcs(AllAI(II),AllAI(II(1)));
Coadd = imProc.stack.coadd(RegAI);
Coadd = imProc.background.background(Coadd); %, 'BackFun',@median, 'BackFunPar',{[1 2]}, 'VarFun',@imUtil.background.rvar);
Coadd = imProc.sources.findMeasureSources(Coadd);
Coadd = imProc.psf.populatePSF(Coadd);
Coadd = imProc.sources.psfFitPhot(Coadd);
[ResAst, Coadd] = imProc.astrometry.astrometryCore(Coadd, 'RA',RA, 'Dec',Dec, 'Scale',PixScale);
Coadd = imProc.astrometry.addCoordinates2catalog(Coadd,'OutUnits','deg');
Coadd.CatData.sortrows('Dec');
Coadd = imProc.calib.photometricZP(Coadd);

FileName = sprintf('Coadd_%s',CurrFilter);
%FITS.write(Coadd.Image, FileName, 'Header',Coadd.Header)

%%
ACpsf = AC.copy;

%%
hist(Nastrom)
hist(AssymRMS,50)

%%

F100 = Nastrom>100;
Edges = (0:0.002:0.1);
X     = (Edges(2:end) + Edges(1:end-1)).*0.5;
N1 = histcounts(AssymRMS(F100), Edges);
N2 = histcounts(AssymRMS(~F100), Edges);
bar(X(:),[N1(:),N2(:)],'stacked')

legend('N_{*}>100','N_{*}<100','Location','NorthEast')

axis([0 0.08 0 100])
H = xlabel('Two axes rms [arcsec]');
H.FontSize = 18;
H.Interpreter = 'latex';

H = ylabel('Number');
H.FontSize = 18;
H.Interpreter = 'latex';

print AssymRMS_Hist.eps -depsc2

%%

CurrFilter = 'i';

% parallactic angle
LST = celestial.time.lst(JD, Long./RAD);
PA  = celestial.coo.parallactic_angle(RA./RAD,Dec./RAD,LST,Lat./RAD);
HA  = celestial.convert.convert_ha(RA, JD, 'InUnits','deg','Long',Long,'LongUnits','deg');


FlagGoodAstrometry = AssymRMS>0.001 & AssymRMS<0.04 & Nastrom>100 & strcmp(Filter,CurrFilter); % & abs(HA)<20;
% number of good images
sum(FlagGoodAstrometry)

HA = HA(FlagGoodAstrometry);
PA = PA(FlagGoodAstrometry);
LST = LST(FlagGoodAstrometry);
FilterGood = Filter(FlagGoodAstrometry);

ACg = AC(FlagGoodAstrometry);
% merge
[MergedCat, MatchedS, ResultSubIm.ResZP, ResultSubIm.ResVar, ResultSubIm.FitMotion] = imProc.match.mergeCatalogs(ACg);    
MS=MatchedS.mergeByCoo(MatchedS(1));
%MS=MatchedS.mergeByCoo(MSi);
MS=MS.addSrcData;

switch CurrFilter
    case 'q'
        MSq = MS.copy;
        MSq = MSq.addExtMagColor;
    case 'i'
        MSi = MS.copy;
        MSi = MSi.addExtMagColor;

end

Dist = celestial.coo.sphere_dist_fast(RA./RAD, Dec./RAD, MS.SrcData.RA./RAD, MS.SrcData.Dec./RAD);
[MinDist,Isn]=min(Dist.*RAD.*3600);

plot(MS.Data.RA(:,Isn), MS.Data.Dec(:,Isn),'O');

[~,AM]=celestial.coo.ha2alt(HA,MS.Data.Dec(:,Isn),Lat,'deg');
MeasuredRA = MS.Data.RA(:,Isn);
MeasuredDec = MS.Data.Dec(:,Isn);
plot(AM.*sin(PA), (MeasuredDec-median(MeasuredDec)).*3600,'.')
plot(AM.*cos(PA), (MeasuredDec-median(MeasuredDec)).*3600,'.')

Nim = sum(FlagGoodAstrometry);

switch CurrFilter
    case 'q'
        RA_Best_q  = median(MS.Data.RA(:,Isn),'all','omitnan');
        Dec_Best_q = median(MS.Data.Dec(:,Isn),'all','omitnan');
        RA_rstd_q  = tools.math.stat.rstd(MS.Data.RA(:,Isn)).*cosd(Dec_Best_q).*3600
        Dec_rstd_q = tools.math.stat.rstd(MS.Data.Dec(:,Isn)).*3600
        
        RA_Err  = RA_rstd_q./sqrt(Nim)
        Dec_Err = Dec_rstd_q./sqrt(Nim)

    case 'i'
        RA_Best_i  = median(MS.Data.RA(:,Isn),'all','omitnan');
        Dec_Best_i = median(MS.Data.Dec(:,Isn),'all','omitnan');
        RA_rstd_i  = tools.math.stat.rstd(MS.Data.RA(:,Isn)).*cosd(Dec_Best_i).*3600
        Dec_rstd_i = tools.math.stat.rstd(MS.Data.Dec(:,Isn)).*3600

        RA_Err  = RA_rstd_i./sqrt(Nim)
        Dec_Err = Dec_rstd_i./sqrt(Nim)
end


MS4 = MS.binning(4, 'CreateNewObj',true);
MS16 = MS.binning(16, 'CreateNewObj',true);
MS64 = MS.binning(64, 'CreateNewObj',true);

% Wird, but the moment position estimation is a few times better than PSF
% fitting - possibly because of undersampling?
MagType = 'MAG_APER_3'
switch CurrFilter
    case 'i'
        MagDiff = nanmedian(MS.SrcData.(MagType)-MSi.SrcData.ExtMag)
    case 'q'
        MagDiff = nanmedian(MS.SrcData.(MagType)-MSq.SrcData.ExtMag)
end
plot(MS.SrcData.(MagType)-MagDiff, 3600.*std(MS.Data.Dec,[],1,'omitnan'),'.');
hold on;
plot(MS4.SrcData.(MagType)-MagDiff, 3600.*std(MS4.Data.Dec,[],1,'omitnan'),'.');
plot(MS16.SrcData.(MagType)-MagDiff, 3600.*std(MS16.Data.Dec,[],1,'omitnan'),'.');
%plot(MS64.SrcData.(MagType), 3600.*std(MS64.Data.Dec,[],1,'omitnan'),'.');

set(gca,'YS','log')
axis([14 21 1e-3 2])

H = xlabel('GAIA Mag');
H.FontSize = 18;
H.Interpreter = 'latex';
H = ylabel(sprintf('Dec. rms (%s-band) [arcsec]',CurrFilter));
H.FontSize = 18;
H.Interpreter = 'latex';

eval(sprintf('print SN2024vjm_rms_vs_mag_1_4_16_%s.eps -depsc2',CurrFilter));

Fi = strcmp(FilterGood,'i');
Fq = strcmp(FilterGood,'q');

%

switch lower(CurrFilter)
    case 'q'
        RA_q  = MeasuredRA;
        Dec_q = MeasuredDec;
        PA_q  = PA;
        AM_q  = AM;
    case 'i'
        RA_i  = MeasuredRA;
        Dec_i = MeasuredDec;
        PA_i  = PA;
        AM_i  = AM;
end


%%

N_i = numel(RA_i);
N_q = numel(RA_q);

RefAM = 0;

H_q = [ones(N_q,1), (AM_q-RefAM).*sin(PA_q), (AM_q-RefAM).*cos(PA_q), zeros(N_q,1), zeros(N_q,1)];
H_i = [ones(N_i,1), zeros(N_i,1), zeros(N_i,1), (AM_i-RefAM).*sin(PA_i), (AM_i-RefAM).*cos(PA_i)];

H = [H_q; H_i];
RA_all  = [RA_q; RA_i];
Dec_all = [Dec_q; Dec_i]; 

Par_RA = H\RA_all
Par_Dec = H\Dec_all

Resid_RA  = RA_all - H*Par_RA;
Resid_Dec = Dec_all - H*Par_Dec;

std(Resid_RA).*3600.*abs(cosd(mean(RA)))
%     0.011
std(Resid_Dec).*3600
%     0.016

% The std of RA_i, Dec_i
std(RA_i).*3600.*abs(cosd(mean(RA_i)))
%     0.013
std(Dec_i).*3600
%     0.020427

% The std of RA_q, Dec_q
std(RA_q).*3600.*abs(cosd(mean(RA_q)))
%     0.011
std(Dec_q).*3600
%     0.016

% best coordinates
Par_RA(1)
Par_Dec(1)

