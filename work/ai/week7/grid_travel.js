function grid_travel(nRows, nCols) {
    if (nRows === 1 && nCols === 1) {
        return 1;
    }
    else if (nRows === 0 || nCols === 0) {
        return 0;
    }
    // travel down travel right
    return grid_travel(nRows - 1, nCols) + grid_travel(nRows, nCols - 1);
}

console.log(grid_travel(3, 2));