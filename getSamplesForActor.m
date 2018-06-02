function samples=getSamplesForActor(actor)
    sample_files=dir(strcat('voxceleb1/voxceleb1_txt/',actor,'/*.wav'));
    num_samples=length(sample_files);
    samples=cell(num_samples,2);
    for i=1:num_samples
        [samples{i,1},samples{i,2}]=audioread(sample_files(i).name);
        samples{i,1}=samples{i,1}(:,1);
    end
end