function result=normalize(vector)
    result=(vector-min(vector))./(max(vector)-min(vector));
end