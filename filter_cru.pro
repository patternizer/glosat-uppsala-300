pro filter_cru,thalf=thalf,tsin=tsin1,tslow=tslow,tshigh=tshigh,nan=nan,$
  weight=weight,truncate=truncate
;
; Uses the CRU Gaussian weighted filter to separate the low and high frequency
; components of a timeseries (or a set of time series of equal length).
;
; thalf          : period of oscillation that is reduced in amplitude by 50%
; tsin           : input timeseries (1D or 2D, if the latter then assume
;                : order is [SERIES,TIME]
; nan            : If this flag is set, missing values in tsin are
;              replaced by the timeseries local mean before filtering, and then
;                  re-replaced by the missing code afterwards (and in the high
;                  and low frequency series too)
; tslow,tshigh   : low and high frequency components
; weight         : optionally return weights used
; truncate       : 0=pad with data and filter right to ends,
;                : 1=truncate filtered series to stop (thalf-1)/4 from end
;                : 2=truncate filtered series to stop (thalf-1)/2 from end
;                : NB, truncate only affects the ends of the series, not
;                : missing data within the series, which can be "truncated"
;                ; by nan=0 (though with less control).  If a series has
;                ; missing data past the start or end of the series already,
;                ; then TRUNCATE will not truncate the actual data values,
;                : but only the values from the ends of the complete series
;                : including missing code.
;
;-----------------------------------------------------------------------------
;
; Define arrays
;
tsin=reform(tsin1)
tssize=size(tsin)
case tssize[0] of
 1: begin
  n1d=1
  nx=1
  nt=tssize[1]
  tsin=reform(tsin,nx,nt)
 end
 2: begin
  n1d=0
  nx=tssize[1]
  nt=tssize[2]
 end
 else: message,'tsin should be 1D or 2D'
endcase
;print,'nt',nt
tslow=fltarr(nx,nt)
tshigh=fltarr(nx,nt)
;
; Compute number of weights required
;
nw=long(thalf/2.5+0.5)
if (nw mod 2) eq 0 then nw=nw+2 else nw=nw+1
if nw le 7 then nw=7
;print,'nw',nw
weight=fltarr(nw)
;
; Compute weights
;
wfactor=-18./(long(thalf)^2)    ; must convert to long to avoid overflow!
wroot=1./sqrt(2.*!pi)
weight[0]=wroot
wsum=weight[0]
for i = 1L , nw-1 do begin
  weight[i]=wroot*exp(wfactor*float(i)*float(i))
  wsum=wsum+2.*weight[i]
endfor
weight=weight/wsum
;
; If required, pad the timeseries with its local mean where values are missing
;
tspad=tsin
if keyword_set(nan) then begin
 for ix = 0 , nx-1 do begin
  misslist=where(finite(tsin[ix,*]) eq nan,nmiss)
  if (nmiss gt 0) and (nmiss lt nt) then begin
    for i = 0 , nmiss-1 do begin
      ele1=(misslist[i]-nw+1) > (0)
      ele2=(misslist[i]+nw-1) < (nt-1)
      locvals=tsin[ix,ele1:ele2]
      locmean=total(locvals,/nan)/float(total(finite(locvals)))
      tspad[ix,misslist[i]]=locmean
    endfor
  endif
 endfor
endif
;
; Extend ends of timeseries by mean from each end
;
nend=nw-1
tspad2=fltarr(nx,nt+2*nend)
meanst=total(tspad[*,0:nend-1],2)/float(nend)
meanen=total(tspad[*,nt-nend:nt-1],2)/float(nend)
for ix = 0 , nx-1 do begin
 tspad2[ix,*]=[replicate(meanst[ix],nend),reform(tspad[ix,*]),replicate(meanen[ix],nend)]
endfor
tspad=tspad2
;
; Apply the filter
;
for i = 0L , nt-1 do begin
  wsum=weight[0]*tspad[*,i+nend]
  for j = 1 , nw-1 do begin
    wsum=wsum+weight[j]*(tspad[*,i+nend-j]+tspad[*,i+nend+j])
  endfor
  tslow[*,i]=wsum[*]
endfor
;
; Compute the residual (high-frequency) component
;
tshigh=tsin-tslow
;
; Insert the missing value if required
;
if keyword_set(nan) then begin
  misslist=where(finite(tsin) eq 0,nmiss)
  if nmiss gt 0 then begin
    tslow[misslist]=!values.f_nan
    tshigh[misslist]=!values.f_nan
  endif
endif
;
; Truncate ends of the filtered series if required
;
if keyword_set(truncate) then begin
  ; Compute number of start and end values to zero
  nend=(thalf-1.)/2.
  if truncate eq 1 then nend=nend/2.
  nend=fix(nend)
  if nend gt 0 then begin
   tshigh[*,0:nend-1]=!values.f_nan
   tshigh[*,nt-nend:*]=!values.f_nan
   tslow[*,0:nend-1]=!values.f_nan
   tslow[*,nt-nend:*]=!values.f_nan
endif
endif
;
if n1d eq 1 then begin
 tslow=reform(tslow)
 tshigh=reform(tshigh)
endif
;
end
