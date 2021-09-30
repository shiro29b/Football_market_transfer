
#include<stdio.h>
#include<limits.h>
int max(int a, int b)
{
    return (a > b)? a : b;
}


int rodcut(int price[],int n)
{
    int value[n+1];
    value[0] = 0;
    int i, j;
    for (i = 1; i<=n; i++)
    {
        int maxval=INT_MIN;
        for (j=0;j<i;j++)
            maxval=max(maxval,price[j]+value[i-j-1]);
        value[i]=maxval;
    }
    return value[n];
}


int main()
{
	int n;
    printf("Enter the size of the array : ");
    scanf("%d",&n);

    int arr[n];
    printf("Enter array elements : ");
    for(int i=0;i<n;i++)
    {
        scanf("%d",&arr[i]);
    }
	printf("Max Value is %d",rodcut(arr,n));

	return 0;
}
