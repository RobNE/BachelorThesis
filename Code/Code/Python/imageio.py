def read_cube(filename):
    import gdal
    print 'Reading ',filename,'...'
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.CreateCopy('', gdal.Open("/Users/rellerkmann/Desktop/Bachelorarbeit/Bachelorarbeit/BachelorThesis/Code/Data/gms_sample/stack.vrt"))
    cube = dataset.ReadAsArray()
    print '...done'
    return cube

def write_cube(filename, cube):
    import osgeo.gdal_array
    import gdal
    print 'Reading ',filename,'...'
    bands, lines, samples  = cube.shape
    GDALType = osgeo.gdal_array.NumericTypeCodeToGDALTypeCode(cube.dtype)
    driver = gdal.GetDriverByName('ENVI')  
    dataset = driver.Create(filename, samples, lines, bands, GDALType)
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(cube[i])
    dataset = None
    print '...done'
    
#    import numpy
#    cube = numpy.random.normal(size=(5, 100, 200))
#    write_cube(r't:\test', cube)
