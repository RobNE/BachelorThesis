package org.apache.flink.quickstart;

import java.io.IOException;

import org.apache.flink.api.common.io.FileInputFormat;
import org.apache.flink.core.io.InputSplit;
import org.apache.flink.core.io.InputSplitAssigner;

public class BSQInputFormat<T> extends FileInputFormat{

	@Override
	public InputSplitAssigner getInputSplitAssigner(InputSplit[] inputSplits) {
		// TODO Auto-generated method stub
		return null;
	}

	public void open(InputSplit split) throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean reachedEnd() throws IOException {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Object nextRecord(Object reuse) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

}
