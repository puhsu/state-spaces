import torch
import torch.nn.functional as F
from einops import rearrange
from pykeops.torch import Genred


_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors


def _c2r(x): return torch.view_as_real(x)
def _r2c(x): return torch.view_as_complex(x)


def cauchy_conj(v, z, w, num=2, denom=2):
    if num == 1:
        expr_num = 'z * ComplexReal(v) - Real2Complex(ComplexReal(v)*ComplexReal(w) + ComplexImag(v)*ComplexImag(w))'
    elif num == 2:
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
    else: raise NotImplementedError

    if denom == 1:
        expr_denom = 'ComplexMult(z-Real2Complex(ComplexReal(w)), z-Real2Complex(ComplexReal(w))) + Real2Complex(Square(ComplexImag(w)))'
    elif denom == 2:
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'
    else: raise NotImplementedError

    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = _c2r(v)
    z = _c2r(z)
    w = _c2r(w)

    r = 2*cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)


def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.
    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        if not done: A_ = A_ @ A_

    assert x.shape[-1] == L

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x


@torch.no_grad()
def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i
    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        if v is None:
            powers = [powers[-1] @ powers[-1]]
        else:
            powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


""" Utilities for computing convolutions.
There are 3 equivalent views:
    1. causal convolution
    2. multiplication of (lower) triangular Toeplitz matrices
    3. polynomial multiplication (mod x^N)
"""


def construct_toeplitz(v, f=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A = Z_f. This uses vectorized indexing and cumprod so it's much
    faster than using the Krylov function.
    Parameters:
        v: the starting vector of size n or (rank, n).
        f: real number
    Returns:
        K: Krylov matrix of size (n, n) or (rank, n, n).
    """
    n  = v.shape[-1]
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, None] + b[None]
    K = v[..., indices]
    K[..., indices < 0] *= f
    return K


def triangular_toeplitz_multiply_(u, v, sum=None):
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)
    uv_f = u_f * v_f
    if sum is not None:
        uv_f = uv_f.sum(dim=sum)
    output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
    return output


def triangular_toeplitz_multiply_padded_(u, v):
    """ Same as triangular_toeplitz_multiply but inputs and output assume to be 0-padded already. """
    n = u.shape[-1]
    assert n % 2 == 0
    u_f = torch.fft.rfft(u, n=n, dim=-1)
    v_f = torch.fft.rfft(v, n=n, dim=-1)
    uv_f = u_f * v_f
    output = torch.fft.irfft(uv_f, n=n, dim=-1)
    output[..., n:] = 0
    return output


class TriangularToeplitzMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return triangular_toeplitz_multiply_(u, v)

    @staticmethod
    def backward(ctx, grad):
        u, v = ctx.saved_tensors
        d_u = triangular_toeplitz_multiply_(grad.flip(-1), v).flip(-1)
        d_v = triangular_toeplitz_multiply_(grad.flip(-1), u).flip(-1)
        return d_u, d_v


class TriangularToeplitzMultFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_expand = F.pad(u, (0, n))
        v_expand = F.pad(v, (0, n))
        u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
        v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad.flip(-1), (0, n))
        g_f = torch.fft.rfft(g_expand, n=2*n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=2*n, dim=-1)[..., :n]
        d_v = torch.fft.irfft(gu_f, n=2*n, dim=-1)[..., :n]
        d_u = d_u.flip(-1)
        d_v = d_v.flip(-1)
        return d_u, d_v


class TriangularToeplitzMultPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        output = triangular_toeplitz_multiply_(u, v)
        return output

    @staticmethod
    def backward(ctx, grad):
        u, v = ctx.saved_tensors
        d_u = triangular_toeplitz_multiply_padded_(grad.flip(-1), v).flip(-1)
        d_v = triangular_toeplitz_multiply_padded_(grad.flip(-1), u).flip(-1)
        return d_u, d_v


class TriangularToeplitzMultPaddedFast(torch.autograd.Function):
    """ Trade off speed (20-25% faster) for more memory (20-25%) """

    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_f = torch.fft.rfft(u, n=n, dim=-1)
        v_f = torch.fft.rfft(v, n=n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=n, dim=-1)
        output[..., n//2:].zero_()
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad[..., :n//2].flip(-1), (0, n//2))
        g_f = torch.fft.rfft(g_expand, n=n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=n, dim=-1)
        d_v = torch.fft.irfft(gu_f, n=n, dim=-1)
        d_u[..., n//2:].zero_()
        d_v[..., n//2:].zero_()
        d_u[..., :n//2] = d_u[..., :n//2].flip(-1) # TODO
        d_v[..., :n//2] = d_v[..., :n//2].flip(-1) # TODO
        return d_u, d_v


# triangular_toeplitz_multiply = triangular_toeplitz_multiply_
triangular_toeplitz_multiply = TriangularToeplitzMult.apply
triangular_toeplitz_multiply_fast = TriangularToeplitzMultFast.apply
triangular_toeplitz_multiply_padded = TriangularToeplitzMultPadded.apply
triangular_toeplitz_multiply_padded_fast = TriangularToeplitzMultPaddedFast.apply


def causal_convolution(u, v, fast=True, pad=False):
    if not pad and not fast:
        return triangular_toeplitz_multiply(u, v)
    if not pad and fast:
        return triangular_toeplitz_multiply_fast(u, v)
    if pad and not fast:
        return triangular_toeplitz_multiply_padded(u, v)
    if pad and fast:
        return triangular_toeplitz_multiply_padded_fast(u, v)


def log_vandermonde(v, x, L, conj=True):
    expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
    vandermonde_mult = Genred(
        expr,
        [
            'v = Vj(2)',
            'x = Vj(2)',
            'l = Vi(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(v, x, l, backend='GPU')
    if conj:
        return 2*_r2c(r).real
    else:
        return _r2c(r)


def log_vandermonde_transpose(u, v, x, L):
    """
    u: ... H L
    v: ... H N
    x: ... H N
    Returns: ... H N
    V = Vandermonde(a, L) : (H N L)
    contract_L(V * u * v)
    """
    expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
    vandermonde_mult = Genred(
        expr,
        [
            'u = Vj(2)',
            'v = Vi(2)',
            'x = Vi(2)',
            'l = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    l = torch.arange(L).to(x)
    u, v, x, l = _broadcast_dims(u, v, x, l)
    u = _c2r(u)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(u, v, x, l, backend='GPU')
    return _r2c(r)