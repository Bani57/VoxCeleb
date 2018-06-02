function tempo=getTempo(sample)
tempo=zeros(1,3);
threshold=0.05*max(sample); 
is_pause=abs(sample)<threshold; % Sound with intensity less than 5% is a pause in speech
tempo(1)=sum(1-is_pause)/sum(is_pause); % Speech to pause ratio
pause_length=0;
num_pauses=0;
first_pause=1;
for i=1:length(is_pause)
    if is_pause(i)==1 && first_pause==1
        first_pause=0;
        pause_length=0;
    elseif is_pause(i)==1 && first_pause==0
        pause_length=pause_length+1;
    elseif is_pause(i)==0 && first_pause==0
        num_pauses=num_pauses+1;
        tempo(3)=tempo(3)+pause_length;
        first_pause=1;
    end
end
tempo(2)=num_pauses; % Number of pauses
tempo(3)=tempo(3)/num_pauses; % Average pause length
end