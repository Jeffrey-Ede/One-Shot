PARENT_DIR = "//Desktop-sa1evjv/f/stills_hq/";
SAVE_DIR = "//Desktop-sa1evjv/h/small_scans-tem/";

k = 0;
for j = 1:1001
    for i = 1:14
        f = PARENT_DIR+"series"+num2str(j)+"/"+num2str(i)+".tif";
        
        img = imread(f);
        shape = size(img);

        side = min(shape);
        img = img(1:side, 1:side);

        small_img = imresize(img, [512, 512], 'method', 'box');
    %     imshow(small_img/max(max(small_img)))
    %     pause
    %     imwrite(small_img, SAVE_DIR+num2str(i)+".tif", "tiff")

        t = Tiff(SAVE_DIR+"/series"+num2str(j)+"/"+num2str(i)+".tif", 'w'); 
        tagstruct.ImageLength = size(small_img, 1); 
        tagstruct.ImageWidth = size(small_img, 2); 
        tagstruct.Compression = Tiff.Compression.None; 
        tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
        tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
        tagstruct.BitsPerSample = 32; 
        tagstruct.SamplesPerPixel = 1; 
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
        t.setTag(tagstruct); 
        t.write(small_img); 
        t.close();
    end
end