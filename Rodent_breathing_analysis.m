%% RODENT BREATHING DATA ANALYSIS
% The objective of the code below is to aid the analysis of rodent
% breathing data. You can look over and manually edit peaks and troughs of
% rodent respiration data and electrophysiological signals of rodent
% medullary neurons, both raw and filtered.

% Initially programmed by Jo Suresh (April 12, 2016) and Tahra Eissa.
% Edited by SungJun Cho for the manual addition of undetected maxima and
% minima of the signal waves. 10/11/2019

clc; clear; close all;
%% Setting Inputs
fprintf('SETTING UP... \n')
tic;

%%%%%%%%%%%%%%%%%%%%%%%%% USER INTERACTIVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FileName          = input('Enter the filename to analyze (program will add .abf extension as defualt):  ','s');
sr                = input('Enter the sampling frequency in Hz:   '); % unit: Hz
tStartTime        = input('Enter START TIME in secs:   ');
tEndTime          = input('Enter END TIME in secs:   ');
minPeakSeparation = input('Enter MIN PEAK SEPARATION in secs (this will be applied to both peaks and valleys):   '); % usually 0.5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% May add minPeakProminence based on the need of user

% Integer operands are required for color operator when used as index.
tStartTime = ceil(tStartTime);
tEndTime = ceil(tEndTime);
%% Read the abf file
FileNameOpen = [FileName '.abf'];
[abfData,si,h] = abfload(FileNameOpen);

% Setting time array
x_raw = abfData(:,1)'; % original signal
t_raw = 1/sr:1/sr:length(x_raw)/sr; % original time array
t = tStartTime+(1/sr):1/sr:tEndTime+(1/sr);

filename = [FileName '_' num2str(tStartTime) 's_' num2str(tEndTime) 's_Results.xlsx'];

% Check whether the file already exists
if exist(filename, 'file')
    error('File with same name already exists. \n')
end

elapsed_set_up = toc;
fprintf('TOOK %.3f SECONDS. \n', elapsed_set_up)
%% Plot Original and Segmented Signals
chans = size(abfData,2); % number of channels
x_timeframed = zeros(chans,length(t));
for k = 1:chans
    if tStartTime == 0
        x_timeframed(k,:) = abfData(1:tEndTime*sr,k)';
    else
        x_timeframed(k,:) = abfData(tStartTime*sr:tEndTime*sr,k)';
    end
end

figure(1);
for k = 1:chans
    subplot(chans,1,k);
    plot(t_raw,abfData(:,k));
    title(['Entier Signal of Channel', num2str(k)])
    xlabel('Time (s)');
    ylabel('Amplitude (AU)')
end

figure(2);
for k = 1:chans
    subplot(chans,1,k);
    plot(t,x_timeframed(k,:));
    xlim([tStartTime, tEndTime]);
    xlabel('Time (s)');
    ylabel('Amplitude (AU)');
    title(['Part of signal being analyzed (' num2str(tStartTime) 's - ' num2str(tEndTime) 's)']);
end

%%%%%%%%%%%%%%%%%%%%%%%%% USER INTERACTIVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
good_chans = input('Which channels would you like to analyze (e.g.[1,2,5])?');
% We want to look at channel 2 and 3: ch2 - integrated breathing; ch3 - raw EEG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if tStartTime == 0
    x_timeframed = abfData(1:tEndTime*sr, good_chans)';
else
    x_timeframed = abfData(tStartTime*sr:tEndTime*sr, good_chans)';
end

fprintf('Please check the figures. Press spacebar to continue. \n')
pause;

%% Signal Filtering
fprintf('Filtering...')
tic;

Ny = sr/2; %Nyquist frequency: half of the sampling rate

%%%%%%%%%%%%%%%%%%%%%%%%% USER INTERACTIVE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% High-pass filter setting in Hz
fhs = input('Enter the high-pass filter setting in Hz:   '); % usually fhs = 0.3
% Low-pass filter setting in Hz
fls = input('Enter the low-pass filter setting in Hz:    '); % usually fls = 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[b1,a1] = butter(2,fls/Ny,'low');
[b2,a2] = butter(2,fhs/Ny,'high');

% Filtering the corresponding signals
xl_filt = zeros(length(good_chans),length(t));
xh_filt = zeros(length(good_chans),length(t));
for k = 1:length(good_chans)
    xl_filt(k,:) = filtfilt(b1,a1,x_timeframed(k,:));
    xh_filt(k,:) = filtfilt(b2,a2,x_timeframed(k,:));
end

Breath_chan = 1;
Breath_chan_raw = 2;
EEG_chan = 3;
EEG_chan_raw = 4;

Breath_chan = find(good_chans==Breath_chan);
EEG_chan = find(good_chans==EEG_chan);
Breath_chan_raw = find(good_chans==Breath_chan_raw);
EEG_chan_raw = find(good_chans==EEG_chan_raw);

% Plotting original and filtered signals
figure(4)
if ~isempty(Breath_chan)
    subplot(4,1,1)
    hold on
    plot(t,x_timeframed(Breath_chan,:),'b');
    plot(t,xl_filt(Breath_chan,:),'r');
    plot(t,xh_filt(Breath_chan,:),'k');
    title('Breathing Data')
    legend('OriginalSignal',['LowPass: ', num2str(fls(1)),'Hz'],['HighPass: ', num2str(fhs(1)),'Hz'],'Location','northeastoutside');
    xlabel('Time in secs');
    ylabel('Amplitude AU');
end
if ~isempty(EEG_chan)
    subplot(4,1,2)
    hold on
    plot(t,x_timeframed(EEG_chan,:),'b');
    plot(t,xl_filt(EEG_chan,:),'r');
    plot(t,xh_filt(EEG_chan,:),'k');
    title('EEG Data')
    legend('OriginalSignal',['LowPass: ', num2str(fls(1)),'Hz'],['HighPass: ', num2str(fhs(1)),'Hz'],'Location','northeastoutside');
    xlabel('Time in secs');
    ylabel('Amplitude AU');
end
if ~isempty(Breath_chan_raw)
    subplot(4,1,3)
    hold on
    plot(t,x_timeframed(Breath_chan_raw,:),'b');
    plot(t,xl_filt(Breath_chan_raw,:),'r');
    plot(t,xh_filt(Breath_chan_raw,:),'k');
    title('Rectified Breathing Data')
    legend('OriginalSignal',['LowPass: ', num2str(fls(1)),'Hz'],['HighPass: ', num2str(fhs(1)),'Hz'],'Location','northeastoutside');
    xlabel('Time in secs');
    ylabel('Amplitude AU');
end
if ~isempty(EEG_chan_raw)
    subplot(4,1,4)
    hold on
    plot(t,x_timeframed(EEG_chan_raw,:),'b');
    plot(t,xl_filt(EEG_chan_raw,:),'r');
    plot(t,xh_filt(EEG_chan_raw,:),'k');
    title('Rectified EEG Data')
    legend('OriginalSignal',['LowPass: ', num2str(fls(1)),'Hz'],['HighPass: ', num2str(fhs(1)),'Hz'],'Location','northeastoutside');
    xlabel('Time in secs');
    ylabel('Amplitude AU');
end

elapsed_filter = toc;
fprintf('took %.3f seconds. \n', elapsed_filter)

fprintf('Please check the figures and press spacebar to continue. \n')
pause;

%% Find Maxima/Peaks
fprintf('Peak detecting ...')
tic;

offset_time = 2; % unit: sec; 
% We need to ignore the first few secs due to the filter artefacts.
xl_filt_offset = xl_filt(:,offset_time*sr:length(xl_filt)); % getting rid of first 2 secs due to initial filter artifact
xh_filt_offset = xh_filt(:,offset_time*sr:length(xh_filt)); % getting rid of first 2 secs due to initial filter artifact
t_offset = t(offset_time*sr:size(xl_filt,2));

locs_l_max = cell(1, length(good_chans));
locs_h_max = cell(1, length(good_chans));
locs_l_min = cell(1, length(good_chans));
locs_h_min = cell(1, length(good_chans));

for k = 1:length(good_chans)    
    [~,locs_l_max{k}] = findpeaks(xl_filt_offset(k,:),1:1:length(xl_filt_offset(k,:)),...
        'Annotate','extents','MinPeakDistance',minPeakSeparation*sr,'MinPeakProminence', 0.001);
    [~,locs_h_max{k}] = findpeaks(xh_filt_offset(k,:),1:1:length(xh_filt_offset(k,:)),...
        'Annotate','extents','MinPeakDistance',minPeakSeparation*sr,'MinPeakProminence',0.001);
    
    xl_filt_offset_inverted = -xl_filt_offset(k,:);
    xh_filt_offset_inverted = -xh_filt_offset(k,:);
    
    [~,locs_l_min{k}] = findpeaks(xl_filt_offset_inverted,1:1:length(xl_filt_offset(k,:)),...
        'MinPeakDistance',minPeakSeparation*sr,'MinPeakProminence',0.001);
    [~,locs_h_min{k}] = findpeaks(xh_filt_offset_inverted,1:1:length(xh_filt_offset(k,:)),...
        'MinPeakDistance',minPeakSeparation*sr,'MinPeakProminence',0.001);
    
    % Minimum distance between peaks is 0.2 secs
end
%% Pull out all the single waves
% We are only going to use all those maxima that are enclosed
% between the first and the last minima.

% Initialize Arrays
cellsize = cell(1,length(good_chans));

final_locs_l_max = cellsize;
final_locs_h_max = cellsize;
final_locs_l_min = cellsize;
final_locs_h_min = cellsize;

FINAL_locs_l_max = cellsize;
FINAL_locs_h_max = cellsize;
wave_LP = cellsize;
wave_HP = cellsize;

wave_rt_LP = cellsize;
wave_ft_LP = cellsize;
FWHM_LP = cellsize;

wave_rt_HP = cellsize;
wave_ft_HP = cellsize;
FWHM_HP = cellsize;

for k = 1:length(good_chans)
    % (1) Isolating the Single Waves
    
    % [1] Low-Pass Filtered Signal: Identifying Peaks
    % Finding the first and last maxima that occur after the first and the
    % last minima, respectively. Wave constricted by these two index points.
    first_idx_l_max = find(locs_l_max{k} > locs_l_min{k}(1),1);
    last_idx_l_max = find(locs_l_max{k} < locs_l_min{k}(length(locs_l_min{k})),1,'last');
    final_locs_l_max_template = locs_l_max{k}(first_idx_l_max:last_idx_l_max);
    
    % [2] High-Pass Filtered Signal: Identifying Peaks
    % Finding the first and last maxima that occur after the first and the
    % last minima, respectively. Wave constricted by these two index points.
    first_idx_h_max = find(locs_h_max{k} > locs_h_min{k}(1),1);
    last_idx_h_max = find(locs_h_max{k} < locs_h_min{k}(length(locs_h_min{k})),1,'last');
    final_locs_h_max_template = locs_h_max{k}(first_idx_h_max:last_idx_h_max);
    
    % (2) Manual Screening
    % Here, the user manually decides to either accept or reject the maxima
    % and minima of the signal wave.
    LP = 'lp'; % low-pass filter
    HP = 'hp'; % high-pass filter
    
    [finalpeaks_LP,finaltrough_LP,ptl_max,ptl_min,~,~] = manualedit_maxmin(xl_filt_offset(k,:), final_locs_l_max_template, locs_l_min{k},LP);
    [finalpeaks_HP,finaltrough_HP,~,~,pth_max,pth_min] = manualedit_maxmin(xh_filt_offset(k,:), final_locs_h_max_template, locs_h_min{k},HP);
    
    % [1] Update manually edited maxima and minima (for detected peaks)
    p = 1;
    for i = 1:length(finalpeaks_LP)
        if finalpeaks_LP(i) == 0 % maxima accepted
            final_locs_l_max{k}(p) = final_locs_l_max_template(i);
            p = p+1;
        end
    end
    
    q = 1;
    for i = 1:length(finaltrough_LP)
        if finaltrough_LP(i) == 0
            final_locs_l_min{k}(q) = locs_l_min{k}(i);
            q = q+1;
        end
    end
    
    p = 1;
    for i = 1:length(finalpeaks_HP)
        if finalpeaks_HP(i) == 0
            final_locs_h_max{k}(p) = final_locs_h_max_template(i);
            p = p+1;
        end
    end
    
    q = 1;
    for i = 1:length(finaltrough_HP)
        if finaltrough_HP(i) == 0
            final_locs_h_min{k}(q) = locs_h_min{k}(i);
            q = q+1;
        end
    end
    
    % [2] Update additionally marked maxima and minima (for undetected peaks)
    % Cartesian coordinates (x,y) on the plot where x=time(s) and y=amplitude(AU)
    
    % IMPORTANT: When marking additional points on the plot, make sure to
    % remember that the user can only mark the points that are between the
    % first and last minimum or maximum (whichever one is first or last).
    
    % NOTE: You should mark the point in the order of maxima and minima.
    % 1. Click on the maxima you want to add. If you right click your last
    % maxima, it will move on to the next part.
    % 2. Click on the minima you want to add. If you right click your last
    % minima, it will move on to the next part.
    % 3. Finally, exclude any errors by clicking red triangles and blue
    % squares.
    % 4. Click "Save selected peaks" if you are done with manual editing.
    % You will be doing this process twice per one channel.
    % Ex) One for LP and one for HP per raw EEG signal.
    
    ptsl_min = cat(2,ptl_min{1,1},ptl_min{1,2});
    ptsl_max = cat(2,ptl_max{1,1},ptl_max{1,2});
    
    ptsh_min = cat(2,pth_min{1,1},pth_min{1,2});
    ptsh_max = cat(2,pth_max{1,1},pth_max{1,2});
    
    % Below may need more optimization %%%%%%%%%%%%%%%%%
    ptsl_min = ceil(ptsl_min);
    ptsl_max = ceil(ptsl_max);
    ptsh_min = ceil(ptsh_min);
    ptsh_max = ceil(ptsh_max);
    
    opt_frame = diff(t_offset);
    opt_frame = ceil(100*opt_frame(1,1)*sr);
    
    opt_loc_l_min = zeros(1,size(ptsl_min,1));
    opt_loc_l_max = zeros(1,size(ptsl_max,1));
    
    opt_loc_h_min = zeros(1,size(ptsh_min,1));
    opt_loc_h_max = zeros(1,size(ptsh_max,1));
    
    for i = 1:size(ptsl_min,1)
        opt_idx_l_min = min(xl_filt_offset(k,ptsl_min(i,1)-opt_frame:ptsl_min(i,1)+opt_frame));
        opt_loc_l_min(1,i) = find(xl_filt_offset(k,:) == opt_idx_l_min);
    end
    
    ptsl_min(:,1) = opt_loc_l_min(1,:);
    
    for i = 1:size(ptsl_max,1)
        opt_idx_l_max = max(xl_filt_offset(k,ptsl_max(i,1)-opt_frame:ptsl_max(i,1)+opt_frame));
        opt_loc_l_max(1,i) = find(xl_filt_offset(k,:) == opt_idx_l_max);
    end
    
    ptsl_max(:,1) = opt_loc_l_max(1,:);
    
    for i = 1:size(ptsh_min,1)
        opt_idx_h_min = min(xh_filt_offset(k,ptsh_min(i,1)-opt_frame:ptsh_min(i,1)+opt_frame));
        opt_loc_h_min(1,i) = find(xh_filt_offset(k,:) == opt_idx_h_min);
    end
    
    ptsh_min(:,1) = opt_loc_h_min(1,:);
    
    for i = 1:size(ptsh_max,1)
        opt_idx_h_max = max(xh_filt_offset(k,ptsh_max(i,1)-opt_frame:ptsh_max(i,1)+opt_frame));
        opt_loc_h_max(1,i) = find(xh_filt_offset(k,:) == opt_idx_h_max);
    end
    
    ptsh_max(:,1) = opt_loc_h_max(1,:);
    
    
    % Case: No addition needed
    noaddition = cell(1,4);
    
    noaddition{1,1} = input('Did you additionally marked peaks by clicking the lowpass signal plot? [Y/N]: ','s'); % for lowpass maxima
    noaddition{1,2} = input('Did you additionally marked troughs by clicking the lowpass signal plot? [Y/N]: ','s'); % for lowpass minima
    noaddition{1,3} = input('Did you additionally marked peaks by clicking the highpass signal plot? [Y/N]: ','s'); % for highpass maxima
    noaddition{1,4} = input('Did you additionally marked troughs by clicking the highpass signal plot? [Y/N]: ','s'); % for hipass minima
    
    for i = 1:length(noaddition)
        switch noaddition{1,i}
            case 'Y'
                continue;
            case 'N'
                if i == 1
                    ptsl_max(:,1) = ptsl_max(:,1)*0;
                elseif i == 2
                    ptsl_min(:,1) = ptsl_min(:,1)*0;
                elseif i == 3
                    ptsh_max(:,1) = ptsh_max(:,1)*0;
                elseif i == 4
                    ptsh_min(:,1) = ptsh_min(:,1)*0;
                end
        end
    end
  
    % Include the points designated on the plot by the user
    % Add designated x-ccordinate indices that represent time
    final_locs_l_max{1,k} = sort([final_locs_l_max{1,k}, ptsl_max(:,1)']);
    final_locs_h_max{1,k} = sort([final_locs_h_max{1,k}, ptsh_max(:,1)']);
    final_locs_l_min{1,k} = sort([final_locs_l_min{1,k}, ptsl_min(:,1)']);
    final_locs_h_min{1,k} = sort([final_locs_h_min{1,k}, ptsh_min(:,1)']);
    
    % Case N: Need to delete zero that has no meaning
    % This method is possible, because we have offsetting 2 seconds.
    final_locs_l_max{1,k} = final_locs_l_max{1,k}(final_locs_l_max{1,k} ~= 0);
    final_locs_h_max{1,k} = final_locs_h_max{1,k}(final_locs_h_max{1,k} ~= 0);
    final_locs_l_min{1,k} = final_locs_l_min{1,k}(final_locs_l_min{1,k} ~= 0);
    final_locs_h_min{1,k} = final_locs_h_min{1,k}(final_locs_h_min{1,k} ~= 0);
    
    figure;
    subplot(2,1,1);
    hold on;
    plot(t_offset,xl_filt_offset(k,:));
    plot((final_locs_l_max{k}./sr) + offset_time + tStartTime, xl_filt_offset(k,final_locs_l_max{k}),'rv','MarkerFaceColor', 'r');
    plot((final_locs_l_min{k}./sr) + offset_time + tStartTime, xl_filt_offset(k,final_locs_l_min{k}),'rs','MarkerFaceColor', 'b');
    grid on;
    legend('Filtered Signal','Maxima','Minima','Location','northeastoutside');
    xlabel('Time (s)')
    ylabel('Amplitude (AU)')
    title(['Low-Pass Filtered Signal with 2s offset: Peaks Detected for Channel', num2str(good_chans(k))]);
    
    subplot(2,1,2);
    hold on;
    plot(t_offset,xh_filt_offset(k,:));
    plot((final_locs_h_max{k}./sr) + offset_time + tStartTime,xh_filt_offset(k,final_locs_h_max{k}),'rv','MarkerFaceColor','r');
    plot((final_locs_h_min{k}./sr) + offset_time + tStartTime,xh_filt_offset(k,final_locs_h_min{k}),'rs','MarkerFaceColor','b');
    grid on;
    legend('Filtered Signal','Maxima','Minima','Location','northeastoutside')
    xlabel('Time (s)')
    ylabel('Amplitude (AU)')
    title(['High-Pass Filtered Signal with 2s offset: Peaks Detected for Channel', num2str(good_chans(k))])
    
    fprintf('Check the figure and press spaceboar to proceed. \n');
    pause;
    
    % [3] Finalizing the edited waves
    % For low-pass filtered signal
    count_l = 1;
    for i = 1: length(final_locs_l_max{k})
        left_idx_l_min = find(final_locs_l_min{k} < final_locs_l_max{k}(i),1,'last');
        right_idx_l_min = find(final_locs_l_min{k} > final_locs_l_max{k}(i),1,'first');
        
        if (xl_filt_offset(k,final_locs_l_min{k}(left_idx_l_min)) >= xl_filt_offset(k,final_locs_l_max{k}(i))) || ...
            (xl_filt_offset(k,final_locs_l_min{k}(right_idx_l_min)) >= xl_filt_offset(k,final_locs_l_max{k}(i)))
            continue; % if true, remaining statements (below) in i-th for-loop is essentially skipped
        end
        
        FINAL_locs_l_max{k}(count_l) = final_locs_l_max{k}(i);
        wave_LP{k}{count_l} = xl_filt_offset(k,final_locs_l_min{k}(left_idx_l_min):final_locs_l_min{k}(right_idx_l_min));
        myWaveTempL = xl_filt_offset(k,final_locs_l_min{k}(left_idx_l_min):final_locs_l_min{k}(right_idx_l_min));
        
        % Risetime and falltime set for the time it takes for the response
        % to rise and fall from 20% to 80% (or vice versa) of the
        % steady-state response, respectively.
        rt_left_LP  = risetime(myWaveTempL,sr,'PercentReferenceLevels',[20 80],'StateLevels',[xl_filt_offset(k,final_locs_l_min{k}(left_idx_l_min)), xl_filt_offset(k,final_locs_l_max{k}(i))]);
        ft_right_LP = falltime(myWaveTempL,sr,'PercentReferenceLevels',[20 80],'StateLevels',[xl_filt_offset(k,final_locs_l_min{k}(right_idx_l_min)), xl_filt_offset(k,final_locs_l_max{k}(i))]);
        
        if length(rt_left_LP) ~= 1
            wave_rt_LP{k}(count_l) = -99;
        else
            wave_rt_LP{k}(count_l) = rt_left_LP;
        end
        
        if length(ft_right_LP) ~= 1
            wave_ft_LP{k}(count_l) = - 99;
        else
            wave_ft_LP{k}(count_l) = ft_right_LP;
        end
        
        % State levels for full width at half maximum (FWHM) are stated as
        % below since there exist cases in which the starting point of
        % upward trace is much lower than the ending point of downward
        % trace.
        
        pw_l = pulsewidth(myWaveTempL,sr,'StateLevels',[max([xl_filt_offset(k,final_locs_l_min{k}(left_idx_l_min)) xl_filt_offset(k,final_locs_l_min{k}(right_idx_l_min))]), xl_filt_offset(k,final_locs_l_max{k}(i))]);
        
        if length(pw_l) ~= 1
            FWHM_LP{k} = - 99;
        else
            FWHM_LP{k} = pw_l;
        end
        
        count_l = count_l + 1;
    end
    
    % For high-pass filtered signal
    count_h = 1;
    
    for i = 1:length(final_locs_h_max{k})
        left_idx_h_min = find(final_locs_h_min{1,k} < final_locs_h_max{1,k}(i),1,'last');
        right_idx_h_min = find(final_locs_h_min{1,k} > final_locs_h_max{1,k}(i),1,'first');
        
        if xh_filt_offset(k,final_locs_h_min{k}(left_idx_h_min)) >= xh_filt_offset(k,final_locs_h_max{k}(i)) ...
                || xh_filt_offset(k,final_locs_h_min{k}(right_idx_h_min)) >= xh_filt_offset(k,final_locs_h_max{k}(i))
            continue;
        end
        
        FINAL_locs_h_max{k}(count_h) = final_locs_h_max{k}(i);
        wave_HP{k}{count_h} = xh_filt_offset(k,final_locs_h_min{k}(left_idx_h_min):final_locs_h_min{k}(right_idx_h_min));
        myWaveTempH = xh_filt_offset(k,final_locs_h_min{k}(left_idx_h_min):final_locs_h_min{k}(right_idx_h_min));
        
        rt_left_HP = risetime(myWaveTempH,sr,'PercentReferenceLevels',[20 80],'StateLevels',[xh_filt_offset(k,final_locs_h_min{k}(left_idx_h_min)), xh_filt_offset(k,final_locs_h_max{k}(i))]);
        ft_right_HP = falltime(myWaveTempH,sr,'PercentReferenceLevels',[20 80],'StateLevels',[xh_filt_offset(k,final_locs_h_min{k}(right_idx_h_min)), xh_filt_offset(k,final_locs_h_max{k}(i))]);
        
        if length(rt_left_HP) ~= 1
            wave_rt_HP{k}(count_h) = -99;
        else
            wave_rt_HP{k}(count_h) = rt_left_HP;
        end
        
        if length(ft_right_HP) ~= 1
            wave_ft_HP{k}(count_h) = -99;
        else
            wave_ft_HP{k}(count_h) = ft_right_HP;
        end
        
        % State levels for full width at half maximum (FWHM) are stated as
        % below since there exist cases in which the starting point of
        % upward trace is much lower than the ending point of downward
        % trace.
        
        pw_h = pulsewidth(myWaveTempH,sr,'StateLevels',[max([xh_filt_offset(k,final_locs_h_min{k}(left_idx_h_min)) xh_filt_offset(k,final_locs_h_min{k}(right_idx_h_min))]), xh_filt_offset(k,final_locs_h_max{k}(i))]);
        
        if length(pw_h) ~= 1
            FWHM_HP{k}(count_h) = -99;
        else
            FWHM_HP{k}(count_h) = pw_h;
        end
        
        count_h = count_h + 1;
    end
    
    elapsed_peak = toc;
    fprintf('took %.3f seconds. \n', elapsed_peak);   
    
    %% Calculating the period and frequency of the detected peaks
    period_LP = cellsize;
    period_HP = cellsize;
    freq_LP   = cellsize;
    freq_HP   = cellsize;
    
    t_max_occurance_l = FINAL_locs_l_max{k}/sr; % transform unit to seconds
    t_max_occurance_h = FINAL_locs_h_max{k}/sr; % transform unit to seconds
    
    period_LP{1,k} = diff(t_max_occurance_l);
    period_HP{1,k} = diff(t_max_occurance_h);
    
    freq_LP{1,k} = 1./period_LP{1,k};
    freq_HP{1,k} = 1./period_HP{1,k};
    %% Measuring scores on the irregularity index
    irreg_score_LP = cellsize;
    irreg_score_HP = cellsize;
    
    if length(period_LP{k}) < 3
        irreg_score_LP{k} = 0;
        warning('Error: Need at least 3 maxima for low-pass signal for irregularity score. \n')
    else
        for i = 1:length(period_LP{k}) - 1
            irreg_score_LP{k}(i) = abs((period_LP{k}(i+1)-period_LP{k}(i))/period_LP{k}(i));
        end
    end
    
    if length(period_HP{k}) < 3
        irreg_score_HP{k} = 0;
        warning('Error: Need at least 3 maxima for high-pass signal for irregularity score. \n')
    else
        for i = 1:length(period_HP{k}) - 1
            irreg_score_HP{k}(i) = abs((period_HP{k}(i+1)-period_HP{k}(i))/period_HP{k}(i));
        end
    end
    %% Writing results in the excel formation
    fprintf('Exporting data to excel...');
    
    header = {'Channel#','Time_of_peaks(LP)','Period(LP)','Freq(LP)','Irreg_Score(LP)','Risetime(LP)','Falltime(LP)','FWHM(LP)'...
        'Time_of_peaks(HP)','Period(HP)','Freq(HP)','Irreg_score(HP)','Risetime(HP)','Falltime(HP)','FWHM(HP)','Sample_reate(Hz)','Filter_freq(LP)','Filter_freq(HP)','MinPeakSeparation'};
    
    filename_chan = [FileName '_' num2str(tStartTime) 's_' num2str(tEndTime) 's_Results_chan_' num2str(good_chans(k)) '.xlsx'];
    
    sheet = k;
    xlswrite(filename,header,sheet);
    
    xlRange = 'A2';
    data = {filename_chan};
    xlswrite(filename,data,sheet,xlRange);
    
    xlRange = 'B2';
    t_correction = offset_time + tStartTime;
    xlswrite(filename,((t_max_occurance_l)+t_correction)',sheet,xlRange);
    
    xlRange = 'C2';
    xlswrite(filename,period_LP{k}',sheet,xlRange);
    
    xlRange = 'D2';
    xlswrite(filename,freq_LP{k}',sheet,xlRange);
    
    xlRange = 'E2';
    xlswrite(filename,irreg_score_LP{k}',sheet,xlRange);
    
    xlRange = 'F2';
    xlswrite(filename,wave_rt_LP{k}',sheet,xlRange);
    
    xlRange = 'G2';
    xlswrite(filename,wave_ft_LP{k}',sheet,xlRange);
    
    xlRange = 'H2';
    xlswrite(filename,FWHM_LP{k}',sheet,xlRange);
    
    xlRange = 'I2';
    xlswrite(filename,((t_max_occurance_h)+t_correction)',sheet,xlRange);
    
    xlRange = 'J2';
    xlswrite(filename,period_HP{k}',sheet,xlRange);
    
    xlRange = 'K2';
    xlswrite(filename,freq_HP{k}',sheet,xlRange);
    
    xlRange = 'L2';
    xlswrite(filename,irreg_score_HP{k}',sheet,xlRange);
    
    xlRange = 'M2';
    xlswrite(filename,wave_rt_HP{k}',sheet,xlRange);
    
    xlRange = 'N2';
    xlswrite(filename,wave_ft_HP{k}',sheet,xlRange);
    
    xlRange = 'O2';
    xlswrite(filename,FWHM_HP{k}',sheet,xlRange);
    
    xlRange = 'P2';
    data2 = {num2str(sr)};
    xlswrite(filename,data2,sheet,xlRange);
    
    xlRange = 'Q2';
    data3 = {num2str(fls)};
    xlswrite(filename,data3,sheet,xlRange);
    
    xlRange = 'R2';
    data4 = {num2str(fhs)};
    xlswrite(filename,data4,sheet,xlRange);
    
    xlRange = 'S2';
    data5 = {num2str(minPeakSeparation)};
    xlswrite(filename,data5,sheet,xlRange);
    
    fprintf('Completed export on excel sheet. \n');
end
