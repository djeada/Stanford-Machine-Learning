## Linear Algebra - review
Matrices are rectangular arrays of numbers written between square brackets and are often represented by capital letters. They are used to organize and index large amounts of data and have dimensions in the form of rows and columns. Vectors are special types of matrices that are usually represented by lower case letters and have only one column. Matrices can be manipulated using operations such as addition, multiplication by a scalar, multiplication by a vector, and multiplication by another matrix. In practice, matrices are often used to apply multiple hypotheses to a dataset, as in the case of predicting house prices. Matrix multiplication has some specific properties, such as the lack of commutativity and the associativity. The identity matrix, denoted by the letter I, is also an important concept in matrix manipulation. The inverse of a matrix is another matrix that can be used to "undo" the effects of matrix multiplication.

## Matrices - overview
* Rectangular array of numbers written between square brackets.
* 2D array.
* Named as capital letters (A,B,X,Y).
* It allows you to organize, index, and access a large amount of data.
* Dimension of a matrix are [Rows x Columns].
* $A_{(i,j)} =$ entry in $i^{th}$ row and $j^{th}$ column.

![matrix_element](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_element.png)

## Vectors - overview
* n by 1 matrix.
* Usually referred to as a lower case letter
* vi is an ith element.
* It allows you to organize, index, and access a large amount of data.
* Dimension of a matrix are [Rows x Columns].
* $A_{(i,j)} =$ entry in $i^{th}$ row and $j^{th}$ column.

$$
  y = \begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots \\
    x_{m}
  \end{bmatrix}
$$

### Matrix manipulation

### Addition

![matrix_addition](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_addition.png)

### Multiplication by scalar

![matrix_mult_scalar](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_mult_scalar.png)

### Multiplication by vector

![matrix_mult_vector](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_mult_vector.png)

### Multiplication by matrix

![matrix_mult_matrix](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_mult_matrix.png)

* Take matrix A and multiply by the first column vector from B.
* Take the matrix A and multiply by the second column vector from B.

![matrix_mult_column](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_mult_column.png)

## Matrices in prcactice

* House prices. However, we now have three hypotheses and the same data set.
* We can use matrix-matrix multiplication to efficiently apply all three hypotheses to all data.
* For example, suppose we have four houses and we wish to guess their prices. There are three opposing hypotheses. Because our hypothesis is only one variable, we convert our data (home sizes) vector into a $4x2$ matrix by adding an extra column of 1s.

![matrix_mult_use](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_mult_use.png)

## Matrix multiplication properties

### Lack of Commutativity
When working with raw numbers/scalars multiplication is commutative:

$$3 \cdot 5 == 5 \cdot 3$$

This is not always true for matrix:

$$A \times B \neq B \times A$$

### Associativity

When working with raw numbers/scalars multiplication is associative:

$$3 \cdot 5 \cdot 2 == 3 \cdot 10 == 15 \cdot 2$$

This also holds true for matrix:

$$A \times (B \times C) ==  (A  \times B) \times C$$

## Identity matrix
When working with raw numbers/scalars 1 is always the identity element:

$$z \cdot 1 = z$$

In matrices we have an identity matrix called I:

$$
I =
  \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
  \end{bmatrix}
$$

If multiplication between matrices A and I is possible:

$$A \times I = A$$

## Matrix inverse
When working with raw numbers/scalars multiplication we can usually take
their inverse:

$$ x \cdot \frac{1}{x} = 1$$

* In the space of real numbers not everything has an inverse. E.g. 0 does not have an inverse!
* The only matrices that have an inverse are square matrices.
* There are numerical methods used for finding the inverses of matrices.

![matrix_inverse](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_inverse.png)

## Matrix transpose

If A is an $m \times n$ matrix B is a transpose of A.
Then B is an $n \times m$ matrix $A_{(i,j)} = B_{(j,i)}$.

![matrix_transpose](https://github.com/djeada/Stanford-Machine-Learning/blob/main/slides/resources/matrix_transpose.png)
