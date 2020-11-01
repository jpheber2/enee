function  playeq(sf,w)
[signal,Fs]=audioread(sf);

%Equalizer design with 5 FIR filter


%filters Design

%low pass 
[b1,a1]=fir1(6,120/(Fs/2),'low');

%first bandpass
[b2,a2]=fir1(6,[50,900]/(Fs/2),'bandpass');

%second bandpass
[b3,a3]=fir1(6,[400,3000]/(Fs/2),'bandpass');

%third bandpass
[b4,a4]=fir1(6,[180,6000]/(Fs/2),'bandpass');

%high pass
[b5,a5]=fir1(6,2500/(Fs/2),'high');


a={a1,a2,a3,a4,a5};
b={b1,b2,b3,b4,b5};

%combine the 5 impulse responses as a weighted sum
%filter the input signal with the weighted impulse response

sigeq=zeros(size(signal));

for i =1:5
    
    sigeq=sigeq+filter(b{i}*w(i),a{i},signal);
    
end


%play the audio produced by equalizer
soundsc(sigeq,Fs)