		public class Analysis {
			public void reduce(Iterable<SlicedTile> values, Collector<Tuple4<Integer, Integer, Integer, String>> out) throws Exception {
		O(1)	System.out.println("The reduce is executed");
		O(1)	Integer band = -1;
		O(1)	Integer slicedTileXPos = -1;
		O(1)	Integer slicedTileYPos = -1;
		O(1)?	HashMap<Tuple2<Integer, Integer>, HashMap<Long, Short>> allPixelTimeSeries = new HashMap<Tuple2<Integer, Integer>, HashMap<Long, Short>>(slicedTileWidth*slicedTileHeight);

		O(ms)	for (int row = 0; row < slicedTileHeight; row++) {
		O(ms)		for (int col = 0; col < slicedTileWidth; col++) {
		O(1)?			HashMap<Long, Short> timeSeries = new HashMap<Long, Short>();
		O(1)			allPixelTimeSeries.put(new Tuple2<Integer, Integer>(col, row), timeSeries);
					}
				}
		O(1)?	ArrayList<SlicedTile> slicedTiles = new ArrayList<SlicedTile>();
O(p*ms^2)O(p)	for (SlicedTile slicedTile : values) {
		O(1)		if (band == -1) {
		O(1)			band = slicedTile.getBand();
		O(1)		}if (slicedTileXPos == -1) {
		O(1)			slicedTileXPos = slicedTile.getPositionInTile().f0;
		O(1)		}if (slicedTileYPos == -1) {
		O(1)			slicedTileYPos = slicedTile.getPositionInTile().f1;
		O(1)		}if (band != slicedTile.getBand()) {
		O(1)			System.out.println("The band should be " + band + " but is: " + slicedTile.getBand());
					}
		O(1)		long acquisitionDate = Long.parseLong(slicedTile.getAqcuisitionDate(), 10);
		O(1)		short[] S16Tile = slicedTile.getSlicedTileS16Tile(); //ATTENTION: Changed from original algo

		O(ms)		for (int row = 0; row < slicedTileHeight; row++) {
		O(ms)			for (int col = 0; col < slicedTileWidth; col++) {
		O(1)				short pixelVegetationIndex = S16Tile[row*slicedTileWidth + col];
		O(1)				Tuple2<Integer, Integer> position = new Tuple2<Integer, Integer>(col, row);
		O(1)				allPixelTimeSeries.get(position).put(acquisitionDate, pixelVegetationIndex);
						}
					}
		O(1)			slicedTiles.add(slicedTile);
				}

		O(1)?	List<Tuple2<Integer, Integer>> allPixelTimeSeriesList = new ArrayList<Tuple2<Integer, Integer>>(allPixelTimeSeries.keySet());

		O(ms^2)	for (Tuple2<Integer, Integer> position : allPixelTimeSeriesList) {
		O(1)?		List<Long>trainingSetList = new ArrayList<Long>(allPixelTimeSeries.get(position).keySet());
		O(1)		int trainingSetSize = trainingSetList.size();
		O(1)		double[] train_x = new double[trainingSetSize];
		O(1)?		svm_node[] predict_x = new svm_node[trainingSetList.size()];
		O(1)		double[] train_y = new double[trainingSetSize];

		O(p)		for (int i=0; i < trainingSetSize; i++) {
		O(1)			Long aqcisitionDate = trainingSetList.get(i);
		O(1)			train_x [i] = aqcisitionDate.doubleValue();
		O(1)			train_y [i] = allPixelTimeSeries.get(position).get(aqcisitionDate);
		O(1)			if (train_y [i] > maxValue) {
		O(1)				maxValue = train_y[i];
						}
					}

		O(p)		for (int i=0; i < train_y.length; i++) {
		O(1)			train_y[i] = train_y[i] / maxValue;
					}

		O(1)		svm_problem prob = new svm_problem();
		O(1)		int countOfDates = train_x.length;
		O(1)		prob.y = new double[countOfDates];
		O(1)		prob.l = countOfDates;
		O(1)		prob.x = new svm_node[countOfDates][];

		O(p)		for (int i = 0; i < countOfDates; i++){
		O(1)			double value = train_y[i];
		O(1)			prob.x[i] = new svm_node[1];
		O(1)			svm_node node = new svm_node();
		O(1)			node.index = 0;
		O(1)			node.value = train_x[i];
		O(1)			prob.x[i][0] = node;
		O(1)			prob.y[i] = value;
		O(1)			predict_x[i] = node;
					}

		O(1)		svm_parameter param = new svm_parameter();
		O(1)		param.C = 1;
		O(1)		param.eps = 0.001;
		O(1)		param.svm_type = svm_parameter.EPSILON_SVR;
		O(1)		param.kernel_type = svm_parameter.RBF;
		O(1)		param.probability = 1;

		O(p)		svm_model model = svm.svm_train(prob, param);
		O(1)		double[][] svrCoefficients= model.sv_coef;
		O(1)		String svrCoefficientsAsString = new String("Vector b: ");
		O(p)		for (double[] dArray: svrCoefficients) {
		O(p)			for (double d: dArray) {
		O(1)				svrCoefficientsAsString += String.valueOf(d);
		O(1)				svrCoefficientsAsString += ", ";
						}
					}
		O(1)		svrCoefficientsAsString = svrCoefficientsAsString.substring(0, svrCoefficientsAsString.length()-2);

		O(1)		System.out.println("The predicted values after OLS: " + svrCoefficientsAsString);

		O(1)		Tuple4<Integer, Integer, Integer, String> pixelTimeSeriesInformation = new Tuple4<Integer, Integer, Integer, String>();
		O(1)		int xPixelValue = this.slicedTileWidth * slicedTileXPos + position.f0;
		O(1)		pixelTimeSeriesInformation.f0 = xPixelValue;
		O(1)		int yPixelValue = this.slicedTileHeight * slicedTileYPos + position.f1;
		O(1)		pixelTimeSeriesInformation.f1 = yPixelValue;
		O(1)		pixelTimeSeriesInformation.f2 = band;
		O(1)		pixelTimeSeriesInformation.f3 = svrCoefficientsAsString;

		O(1)		out.collect(pixelTimeSeriesInformation);
				}
			}
		}
