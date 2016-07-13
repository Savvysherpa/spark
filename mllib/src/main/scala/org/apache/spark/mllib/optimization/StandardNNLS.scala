package org.apache.spark.mllib.optimization

import java.{util => ju}

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

import scala.math
import scala.util.control.Breaks.{breakable, break}

object StandardNNLS {
  def solve(ata: Array[Double], atb: Array[Double]): Array[Double] = {
    def getNonZeroIndex(Arr: Array[Int]): Array[Int] = {
      Arr.zipWithIndex.filter(_._1 != 0).map(_._2)
    }

    def toVecIndex(i: Int, j: Int) = j*(j - 1)/2 + i - 1

    def fillAtA(ata: Array[Double]): Array[Double] = {
      val triK = ata.size
      val r = ((math.sqrt(8*triK + 1) - 1)/2).toInt
      var res = Array.fill[Double](r*r)(0)
      for (i <- 0 to r - 1; j <- i to r - 1){
        var v = ata(toVecIndex(i + 1, j + 1))
        res(i + r*j) = v
        res(i*r + j) = v
      }
      res
    }

    def solveLLS(AtA: Array[Double], Atb: Array[Double]): Array[Double] = {
      val r = Atb.size
      val info = new intW(0)
      lapack.dppsv("U", r, 1, AtA, Atb, r, info)
      if (info.`val` != 0){
        val AtAmat = fillAtA(AtA)
        val jpvt = Array.fill[Int](r)(0)
        val rcond = 1e-5
        val rank = new intW(r)
        val lwork = 5 * r
        val work = Array.fill[Double](lwork)(0)
        lapack.dgelsy(r, r, 1, AtAmat, r, Atb, r, jpvt, rcond, rank, work, lwork, info)
      }
      Atb
    }

    def solveSubmatrix(ata: Array[Double], atb: Array[Double], PP: Array[Int]): Array[Double] = {
      val r = atb.size
      val lenp = PP.size
      val atb_sub = PP.map(atb(_))
      var ata_sub = new Array[Double]((lenp*( lenp + 1 ))/2)
      for (i <- 1 to lenp; j <- i to lenp){
        ata_sub(toVecIndex(i, j)) = ata(toVecIndex(PP(i - 1) + 1, PP(j - 1) + 1))
      }
      var z = Array.fill[Double](r)(0)
      var zp = solveLLS(ata_sub, atb_sub)
      for (i <- 0 to PP.size - 1) {z(PP(i)) = zp(i)}
      z
    }

    def getGrad(ata: Array[Double], atb: Array[Double], x: Array[Double]): Array[Double] = {
      val r = x.size
      var w = Array.fill[Double](r)(0)
      blas.dspmv("U", r, - 1.0, ata, x.clone, 1, 1.0, w, 1)
      blas.daxpy(w.size, 1.0, atb, 1, w, 1)
      w
    }

    val eps = 2.22E-16 // value copied from Matlab's lsqnonneg
    val r = atb.size // rank
    val tol = 1E-12 // tolerance value

    var x = Array.fill[Double](r)(0) // initialize x as zero vector
    var w = atb.clone

    var P = Array.fill[Int](r)(0) // initial passive set
    var Z = (1 to r).toArray // initial active set
    var ZZ = (0 to r - 1).toArray

    var outerIter = 0 // counts number of iterations
    var itmax = 3*r
    var it = 0 // inner iter

    breakable { while (Z.reduce(_+_) > 0 && ZZ.map(w(_)).reduceLeft(_ max _) > tol) {
      outerIter += 1
      var argmax = 0.0
      var t = 0
      for (i <- ZZ){
	if (argmax < w(i)) { argmax = w(i); t = i
      }}

      t = Z(t)
      P(t - 1) = t
      Z(t - 1) = 0

      // it's possible to just store either PP or ZZ
      var PP = getNonZeroIndex(P) // passive set indices
      ZZ = getNonZeroIndex(Z) // active set indices
      var z = solveSubmatrix(ata, atb, PP)

      breakable { while(PP.map(z(_)).reduceLeft(_ min _) <= tol){
        it += 1
	if(it > itmax){ break }
	var QQ = (0 to r - 1).toArray.filter(i => ((z(i) < tol) && (P(i) != 0)))
	val alpha = QQ.map(i => x(i) / (x(i) - z(i))).reduceLeft(_ min _)
	for(i <- 0 to x.size - 1){ x(i) += alpha*(z(i) - x(i)) }
	for(i <- 0 to r - 1){
	  if((math.abs(x(i)) < tol) && (P(i) != 0)){
	    Z(i) = i + 1
	    P(i) = 0
	}}

	PP = getNonZeroIndex(P)
	ZZ = getNonZeroIndex(Z)
	z = solveSubmatrix(ata, atb, PP)
      }}

      x = z.clone
      w = getGrad(ata, atb, x)
      if(outerIter > itmax){ break }
    }}
    x
   }
}
