package org.apache.flink.examples.java.spatial;

import java.util.Arrays;

import org.apache.commons.lang.ArrayUtils;

public class S16ArrayTest {
	
	/*
	 * Original:
	 * 0  1 | 2  3 | 4  5
	 * 6  7 | 8  9 | 10 11
	 * -------------------
	 * 12 13| 14 15| 16 17				row=3, col=0, rowInSlicedTile=0 RESULT: 12  2*6+0*2=12; 2*6+1*2=14; 2*6+2*2=16
	 * 18 19| 20 21| 22 23				row=3, col=0, rowInSlicedTile=0 RESULT: 18  3*6+0*2=18; 3*6+1*2=20; 3*6+2*2=22
	 * -------------------
	 * 24 25| 26 27| 28 29
	 * 30 31| 32 33| 34 35
	 * 
	 * S16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
	 * 
	 * Gesucht:
	 * 1. Quadrant mit Kantenlänge 2
	 * Position in S16: stelle[0, 1, 4, 5]  Formel: [(row*x_pixel_original) bis (row*x_pixel_original)+x_pixel_new for every row]
	 * 
	 */

	public static void main(String[] args) {
		short[] originalSlicedTileS16Tile = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
				17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};

		//The slicedTiles' height/width in pixels
		int slicedTileHeight = 2;
		int slicedTileWidth = 2;
		//The original tiles height/width in pixels
		int originalTileHeight = 6;
		int originalTileWidth = 6;
		int slicedTilesPerRow = originalTileWidth / slicedTileWidth;
		int slicedTilesPerCol = originalTileHeight / slicedTileHeight;
		
		for (int row = 0; row < slicedTilesPerRow; row++) {
			for (int col = 0; col < slicedTilesPerCol; col++) {
				short[] newSlicedTileS16Tile = new short[0];
				for (int slicedTileRow = 0; slicedTileRow < slicedTileHeight; slicedTileRow++) {
					//System.out.println((row+slicedTileRow)*originalTileWidth+slicedTileWidth*col);
					//System.out.println(row*originalTileWidth+slicedTileRow*slicedTileWidth);
					short[] tempNewSlicedTileS16Tile = Arrays.copyOfRange(originalSlicedTileS16Tile, 
							(row+slicedTileRow)*originalTileWidth+slicedTileWidth*col, 
							(row+slicedTileRow)*originalTileWidth+slicedTileWidth*col+slicedTileWidth);
					newSlicedTileS16Tile = ArrayUtils.addAll(newSlicedTileS16Tile, tempNewSlicedTileS16Tile);
				}
				System.out.println("THis is a new tile with the lenght: " + newSlicedTileS16Tile.length);
				for (int i=0; i < newSlicedTileS16Tile.length; i++) {
					System.out.print(newSlicedTileS16Tile[i] + ", ");
				}
				System.out.println("");
				
			}	
		}
	}
}
